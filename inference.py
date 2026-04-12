"""
Inference Script — Airport Flight Scheduling Environment
=========================================================
Baseline agent using OpenAI-compatible client.

Required env vars:
  API_BASE_URL  — LLM endpoint (e.g. https://router.huggingface.co/v1)
  MODEL_NAME    — model identifier
  HF_TOKEN      — API key / HF token
"""
from __future__ import annotations

import os
import json
import time
import textwrap
from typing import Any, Dict, List

from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf_placeholder")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
MAX_STEPS    = 20
TEMPERATURE  = 0.2
MAX_TOKENS   = 400

# ── Env HTTP client ────────────────────────────────────────────────────────────

def env_post(path: str, payload: Dict = None) -> Dict:
    import urllib.request
    url = f"{ENV_BASE_URL}{path}"
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def env_get(path: str) -> Dict:
    import urllib.request
    url = f"{ENV_BASE_URL}{path}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


# ── Prompt builders ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI Air Traffic Controller managing an airport.
Each step you must output EXACTLY one JSON action object.

Available action types:
- assign_runway : {"action_type":"assign_runway","flight_id":"AA101","runway_id":"R1"}
- assign_gate   : {"action_type":"assign_gate","flight_id":"AA101","gate_id":"G1"}
- delay_flight  : {"action_type":"delay_flight","flight_id":"AA101","delay_minutes":5}
- divert_flight : {"action_type":"divert_flight","flight_id":"AA101"}
- clear_runway  : {"action_type":"clear_runway","flight_id":"AA101","runway_id":"R1"}
- declare_emergency: {"action_type":"declare_emergency","flight_id":"AA101"}

Rules:
1. Never assign a flight to an OCCUPIED or CLOSED runway.
2. Respect wake turbulence separation (heavy before light = 5 min gap).
3. Match gate type: international flights → international gates (I*), domestic → domestic (D* or G*).
4. Declare emergency for any flight with EMERGENCY status immediately.
5. Output ONLY valid JSON. No explanation.
""").strip()

def obs_to_prompt(obs: Dict, step: int) -> str:
    clock = obs.get("clock", 0)
    events = obs.get("recent_events", [])
    flights = obs.get("flights", [])
    runways = obs.get("runways", [])
    gates   = obs.get("gates", [])
    weather = obs.get("weather", {})

    flight_lines = []
    for f in flights:
        if f["status"] in ("landed", "diverted", "departed"):
            continue
        flight_lines.append(
            f"  {f['flight_id']} | {f['weight_class']:6} | {f['status']:10} | "
            f"ETA:{f['eta_minutes']:.0f} | fuel:{f['fuel_remaining_min']:.0f} | "
            f"intl:{f['is_international']} | rwy:{f.get('assigned_runway','—')} | gate:{f.get('assigned_gate','—')}"
        )

    rwy_lines = [
        f"  {r['runway_id']} ({r['name']}) — {r['status']} | avail@{r['available_at']:.0f}"
        for r in runways
    ]
    gate_lines = [
        f"  {g['gate_id']} | {'INTL' if g['is_international'] else 'DOM '} | {'BUSY' if g['occupied'] else 'FREE'}"
        for g in gates
    ]

    return textwrap.dedent(f"""
Step: {step} | Clock: {clock:.0f} min
Weather: {weather.get('condition','?')} | vis:{weather.get('visibility_km','?')}km | wind:{weather.get('wind_speed_kts','?')}kts
Events: {'; '.join(events) if events else 'none'}

ACTIVE FLIGHTS:
{chr(10).join(flight_lines) if flight_lines else '  (none)'}

RUNWAYS:
{chr(10).join(rwy_lines)}

GATES:
{chr(10).join(gate_lines)}

Output ONE JSON action:""").strip()


# ── Main ───────────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str) -> float:
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    obs_data = env_post("/reset", {"task_id": task_id, "seed": 42})
    print(f"  Flights: {len(obs_data.get('flights', []))}")

    for step in range(1, MAX_STEPS + 1):
        if obs_data.get("episode_done") or obs_data.get("done"):
            break

        prompt = obs_to_prompt(obs_data, step)
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = completion.choices[0].message.content or "{}"
            # Strip markdown fences if present
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            action = json.loads(raw)
        except Exception as exc:
            print(f"  Step {step}: LLM error ({exc}), using noop delay")
            # Find first active flight as fallback
            active = [f for f in obs_data.get("flights", [])
                      if f["status"] not in ("landed","diverted","departed")]
            if active:
                action = {"action_type": "delay_flight",
                          "flight_id": active[0]["flight_id"],
                          "delay_minutes": 2}
            else:
                break

        try:
            result = env_post("/step", {"action": action, "task_id": task_id})
        except Exception as exc:
            print(f"  Step {step}: env error ({exc})")
            break

        obs_data = result.get("observation", obs_data)
        reward   = result.get("reward", {})
        done     = result.get("done", False)
        print(f"  Step {step:2d}: {action.get('action_type','?'):20s} | "
              f"reward: {reward.get('value', 0):+.2f} | "
              f"cumulative: {obs_data.get('score_so_far', 0):+.2f}")

        if done:
            print(f"  Episode done at step {step}")
            break

    grade_data = env_get(f"/grade?task_id={task_id}")
    score = grade_data.get("score", 0.0)
    print(f"  Final grade: {score:.4f}")
    return score


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = [
        "single_runway_landing",
        "multi_runway_gate_assignment",
        "storm_disruption_recovery",
    ]

    results = {}
    for task_id in tasks:
        try:
            score = run_task(client, task_id)
            results[task_id] = score
        except Exception as exc:
            print(f"  ERROR on {task_id}: {exc}")
            results[task_id] = 0.0
        time.sleep(1)

    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    for task_id, score in results.items():
        print(f"  {task_id:<40} {score:.4f}")
    avg = sum(results.values()) / len(results)
    print(f"  {'AVERAGE':<40} {avg:.4f}")


if __name__ == "__main__":
    main()
