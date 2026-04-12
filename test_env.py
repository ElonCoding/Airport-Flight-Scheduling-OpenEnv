#!/usr/bin/env python3
"""
test_env.py — Local validation of Airport Scheduling Environment
Run: python test_env.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from airport_env import (
        AirportEnv, AirportAction, ActionType,
        FlightStatus, RunwayStatus, WeightClass
    )
    print("✅ Import OK")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_task(task_id: str, verbose: bool = True):
    print(f"\n{'─'*50}")
    print(f"Testing: {task_id}")
    print(f"{'─'*50}")

    env = AirportEnv(task_id=task_id, seed=42)
    obs = env.reset()

    assert obs.clock == 0.0, "Clock should start at 0"
    assert len(obs.flights) > 0, "Should have flights"
    assert len(obs.runways) > 0, "Should have runways"
    assert len(obs.gates) > 0, "Should have gates"
    print(f"  reset()  → {len(obs.flights)} flights, {len(obs.runways)} runways, {len(obs.gates)} gates")

    # state() returns same obs
    s = env.state()
    assert s.clock == obs.clock, "state() clock mismatch"
    print(f"  state()  → clock={s.clock}")

    # Run steps
    total_reward = 0.0
    for i in range(min(15, len(obs.flights) * 3)):
        # Find an actionable flight
        flight = next(
            (f for f in obs.flights if f.status in ('scheduled', 'holding', 'landing', 'emergency')),
            None
        )
        if flight is None:
            break

        # Choose action based on status
        if flight.status == 'emergency':
            action = AirportAction(action_type=ActionType.DECLARE_EMERGENCY, flight_id=flight.flight_id)
        elif flight.status == 'landing' and not flight.assigned_gate:
            gate = next((g for g in obs.gates if not g.occupied and g.is_international == flight.is_international), None)
            if gate:
                action = AirportAction(action_type=ActionType.ASSIGN_GATE, flight_id=flight.flight_id, gate_id=gate.gate_id)
            else:
                action = AirportAction(action_type=ActionType.DELAY_FLIGHT, flight_id=flight.flight_id, delay_minutes=3)
        else:
            rwy = next((r for r in obs.runways if r.status == 'clear'), None)
            if rwy:
                action = AirportAction(action_type=ActionType.ASSIGN_RUNWAY, flight_id=flight.flight_id, runway_id=rwy.runway_id)
                # Immediately clear it on next step
            else:
                occ_rwy = next((r for r in obs.runways if r.status == 'occupied'), None)
                if occ_rwy:
                    action = AirportAction(action_type=ActionType.CLEAR_RUNWAY, flight_id=flight.flight_id, runway_id=occ_rwy.runway_id)
                else:
                    action = AirportAction(action_type=ActionType.DELAY_FLIGHT, flight_id=flight.flight_id, delay_minutes=2)

        obs, reward, done, info = env.step(action)
        total_reward += reward.value

        if verbose:
            print(f"  step {i+1:2d}: {action.action_type.value:20s} {flight.flight_id} → reward={reward.value:+.2f} | cum={total_reward:+.2f}")

        if done:
            print(f"  Episode done at step {i+1}")
            break

    grade = env.grade()
    assert 0.0 <= grade <= 1.0, f"Grade out of range: {grade}"
    print(f"  grade()  → {grade:.4f}")
    print(f"  total reward: {total_reward:.2f}")
    return grade


def test_all():
    print("\n" + "="*50)
    print("Airport Scheduling Env — Test Suite")
    print("="*50)

    grades = {}
    for task_id in AirportEnv.TASK_IDS:
        try:
            g = test_task(task_id)
            grades[task_id] = g
            assert 0.0 <= g <= 1.0
            print(f"  ✅ PASS")
        except AssertionError as e:
            print(f"  ❌ FAIL: {e}")
            grades[task_id] = -1.0

    print(f"\n{'='*50}")
    print("Results:")
    for task_id, g in grades.items():
        status = "✅" if g >= 0 else "❌"
        print(f"  {status} {task_id:<42} {g:.4f}")

    all_pass = all(g >= 0 for g in grades.values())
    print(f"\n{'✅ All tests passed!' if all_pass else '❌ Some tests failed.'}")
    return all_pass


if __name__ == "__main__":
    ok = test_all()
    sys.exit(0 if ok else 1)
