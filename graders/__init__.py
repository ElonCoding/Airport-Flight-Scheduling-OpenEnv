"""
Task graders for Airport Scheduling Environment.
Each grader returns a float in [0.0, 1.0].
"""
from __future__ import annotations
from typing import List
from airport_env import AirportObservation, FlightStatus


def grade_single_runway_landing(obs: AirportObservation) -> float:
    total = len(obs.flights)
    landed = sum(1 for f in obs.flights if f.status == FlightStatus.LANDED)
    avg_delay = (
        sum(f.delay_minutes for f in obs.flights if f.status == FlightStatus.LANDED)
        / max(landed, 1)
    )
    conflict_free = all(f.assigned_runway for f in obs.flights if f.status == FlightStatus.LANDED)
    score = (
        0.50 * (landed / max(total, 1)) +
        0.35 * max(0.0, 1.0 - avg_delay / 10.0) +
        0.15 * (1.0 if conflict_free else 0.5)
    )
    return round(min(max(score, 0.0), 1.0), 4)


def grade_multi_runway_gate_assignment(obs: AirportObservation) -> float:
    total = len(obs.flights)
    landed = sum(1 for f in obs.flights if f.status == FlightStatus.LANDED)
    gate_assigned = sum(1 for f in obs.flights if f.assigned_gate is not None)
    avg_delay = (
        sum(f.delay_minutes for f in obs.flights if f.status == FlightStatus.LANDED)
        / max(landed, 1)
    )
    score = (
        0.35 * (landed / max(total, 1)) +
        0.30 * (gate_assigned / max(total, 1)) +
        0.35 * max(0.0, 1.0 - avg_delay / 15.0)
    )
    return round(min(max(score, 0.0), 1.0), 4)


def grade_storm_disruption_recovery(obs: AirportObservation) -> float:
    total = len(obs.flights)
    landed = sum(1 for f in obs.flights if f.status == FlightStatus.LANDED)
    diverted = sum(1 for f in obs.flights if f.status == FlightStatus.DIVERTED)
    avg_delay = (
        sum(f.delay_minutes for f in obs.flights if f.status == FlightStatus.LANDED)
        / max(landed, 1)
    )
    # Diverted is sometimes necessary — penalize only excess diversions
    diversion_penalty = max(0, diverted - 3) * 0.05
    score = (
        0.40 * (landed / max(total, 1)) +
        0.25 * max(0.0, 1.0 - avg_delay / 20.0) +
        0.20 * max(0.0, 1.0 - diversion_penalty) +
        0.15 * (landed / max(total - diverted, 1))
    )
    return round(min(max(score, 0.0), 1.0), 4)


GRADERS = {
    "single_runway_landing": grade_single_runway_landing,
    "multi_runway_gate_assignment": grade_multi_runway_gate_assignment,
    "storm_disruption_recovery": grade_storm_disruption_recovery,
}


def grade(task_id: str, obs: AirportObservation) -> float:
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task_id}")
    return grader(obs)
