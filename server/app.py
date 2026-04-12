"""
Airport Scheduling Environment — FastAPI Server
"""
from __future__ import annotations

import os
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from airport_env import (
    AirportEnv, AirportAction, AirportObservation, AirportReward,
    ActionType
)

app = FastAPI(
    title="Airport Scheduling OpenEnv",
    description="AI agent environment for airport runway, gate, and flight scheduling",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global env instances per task ──────────────────────────────────────────────
_envs: Dict[str, AirportEnv] = {}
_active_task: str = "single_runway_landing"

def _get_env(task_id: str = None) -> AirportEnv:
    tid = task_id or _active_task
    if tid not in _envs:
        _envs[tid] = AirportEnv(task_id=tid)
    return _envs[tid]


# ── Request models ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "single_runway_landing"
    seed: int = 42

class StepRequest(BaseModel):
    action: Dict[str, Any]
    task_id: str = "single_runway_landing"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "airport-scheduling-env",
        "version": "0.1.0",
        "tasks": AirportEnv.TASK_IDS,
        "endpoints": ["/reset", "/step", "/state", "/grade", "/tasks", "/health"],
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(req: ResetRequest = None):
    req = req or ResetRequest()
    env = AirportEnv(task_id=req.task_id, seed=req.seed)
    _envs[req.task_id] = env
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(req: StepRequest):
    env = _get_env(req.task_id)
    if env._state is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    try:
        action = AirportAction(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }

@app.get("/state")
def state(task_id: str = "single_runway_landing"):
    env = _get_env(task_id)
    if env._state is None:
        return _get_env(task_id).reset().model_dump()
    return env.state().model_dump()

@app.get("/grade")
def grade(task_id: str = "single_runway_landing"):
    env = _get_env(task_id)
    score = env.grade()
    return {"task_id": task_id, "score": score}

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "id": "single_runway_landing",
                "name": "Single Runway Landing Sequence",
                "difficulty": "easy",
                "description": "Sequence 5 arriving flights onto a single runway minimizing delays.",
            },
            {
                "id": "multi_runway_gate_assignment",
                "name": "Multi-Runway Gate Assignment",
                "difficulty": "medium",
                "description": "Manage 2 runways and 8 gates for 10 simultaneous arrivals.",
            },
            {
                "id": "storm_disruption_recovery",
                "name": "Storm Disruption Recovery",
                "difficulty": "hard",
                "description": "Handle runway closure, emergencies and reassignments during a storm.",
            },
        ]
    }


def start():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    start()
