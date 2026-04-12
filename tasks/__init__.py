"""Task definitions for Airport Scheduling Environment."""
from __future__ import annotations

TASKS = [
    {
        "id": "single_runway_landing",
        "name": "Single Runway Landing Sequence",
        "difficulty": "easy",
        "description": (
            "Sequence 5 arriving flights onto a single runway. "
            "Minimize total delay while respecting ICAO wake turbulence separation rules."
        ),
        "n_flights": 5,
        "n_runways": 1,
        "n_gates": 5,
        "episode_horizon": 30,
    },
    {
        "id": "multi_runway_gate_assignment",
        "name": "Multi-Runway Gate Assignment",
        "difficulty": "medium",
        "description": (
            "Manage 2 runways and 8 gates for 10 simultaneous arrivals and departures. "
            "Assign each flight to the correct runway and gate type."
        ),
        "n_flights": 10,
        "n_runways": 2,
        "n_gates": 8,
        "episode_horizon": 60,
    },
    {
        "id": "storm_disruption_recovery",
        "name": "Storm Disruption Recovery",
        "difficulty": "hard",
        "description": (
            "A severe storm closes runway 2 at T+20. "
            "Reassign all affected flights, handle fuel emergencies, "
            "and restore schedule adherence."
        ),
        "n_flights": 15,
        "n_runways": 2,
        "n_gates": 10,
        "episode_horizon": 90,
    },
]
