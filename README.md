# Airport Flight Scheduling Environment

> **OpenEnv submission** — Real-world AI agent environment for airport operations

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://openenv.dev)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)

---

## Overview

The **Airport Flight Scheduling Environment** simulates the real-world task of Air Traffic Control (ATC) and airport operations management. An AI agent must:

- Sequence arriving flights onto available runways
- Respect wake turbulence separation rules (ICAO standards)
- Assign appropriate gates (domestic vs. international)
- Handle weather disruptions and runway closures
- Manage fuel emergencies with priority overrides
- Minimize total delay minutes across the episode

This is a genuine, high-stakes logistics problem. Airlines and airports spend billions optimizing these exact decisions.

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `clock` | float | Simulated minutes elapsed since episode start |
| `flights` | List[Flight] | All flights with ETA, weight class, fuel state, status |
| `runways` | List[Runway] | Availability, current occupant, available-at time |
| `gates` | List[Gate] | Occupancy, terminal, domestic/international flag |
| `weather` | Weather | Visibility, wind, active NOTAMs |
| `recent_events` | List[str] | Last 6 notable events for context |
| `score_so_far` | float | Cumulative reward |

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `assign_runway` | flight_id, runway_id | Clear flight to land on runway |
| `assign_gate` | flight_id, gate_id | Assign gate post-landing |
| `delay_flight` | flight_id, delay_minutes | Push ETA back by N minutes |
| `divert_flight` | flight_id | Send to alternate airport |
| `clear_runway` | flight_id, runway_id | Mark runway clear after use |
| `declare_emergency` | flight_id | Grant priority to fuel-critical flight |

---

## Tasks

### Task 1: Single Runway Landing Sequence (Easy)
- **Flights:** 5 arriving aircraft
- **Runways:** 1
- **Challenge:** Order flights correctly by weight class to minimize separation penalties
- **Expected score:** 0.70–0.90

### Task 2: Multi-Runway Gate Assignment (Medium)
- **Flights:** 10 mixed arrivals
- **Runways:** 2 | **Gates:** 8 (4 domestic, 4 international)
- **Challenge:** Match gate types, balance runway load, handle simultaneous arrivals
- **Expected score:** 0.50–0.75

### Task 3: Storm Disruption Recovery (Hard)
- **Flights:** 15 flights including 4 with low fuel
- **Event:** Runway 2 closes at t=20 due to storm
- **Challenge:** Reassign all R2 flights, handle emergencies, minimize diversions
- **Expected score:** 0.30–0.60

---

## Reward Function

```
Per-step:   −0.1 × count(overdue unhandled flights)
Landing:    +1.0 per safe landing, +0.3 bonus if on-time (<2 min delay)
Gate:       +0.5 per gate assigned, +0.5 bonus if within 5 min of landing
Conflict:   −5.0 per runway incursion or gate double-booking
Emergency:  +3.0 per fuel emergency resolved before critical
Diversion:  −2.0 avoidable diversion | −0.5 unavoidable
Episode:    Up to +10.0 based on throughput × on-time ratio − conflicts
```

---

## Setup & Usage

### Local Development

```bash
git clone <your-repo>
cd airport-env
pip install -e .

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run baseline
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_your_token
python inference.py
```

### Docker

```bash
docker build -t airport-env .
docker run -p 7860:7860 \
  -e API_BASE_URL=$API_BASE_URL \
  -e MODEL_NAME=$MODEL_NAME \
  -e HF_TOKEN=$HF_TOKEN \
  airport-env
```

### API Endpoints

```
POST /reset   {"task_id": "single_runway_landing", "seed": 42}
POST /step    {"action": {...}, "task_id": "..."}
GET  /state   ?task_id=...
GET  /grade   ?task_id=...
GET  /tasks
GET  /health
```

---

## Baseline Scores

| Task | Agent | Score |
|------|-------|-------|
| single_runway_landing | Qwen2.5-72B | 0.74 |
| multi_runway_gate_assignment | Qwen2.5-72B | 0.58 |
| storm_disruption_recovery | Qwen2.5-72B | 0.41 |

---

## Validation

```bash
openenv validate
```

All checks should pass: typed models, step/reset/state endpoints, task graders, openenv.yaml.
