"""
Airport Flight Scheduling Environment — Core Models & Logic
"""
from __future__ import annotations

import random
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ─── Enums ───────────────────────────────────────────────────────────────────

class WeightClass(str, Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    SUPER = "super"

class FlightStatus(str, Enum):
    SCHEDULED = "scheduled"
    HOLDING = "holding"
    LANDING = "landing"
    LANDED = "landed"
    DEPARTING = "departing"
    DEPARTED = "departed"
    DIVERTED = "diverted"
    EMERGENCY = "emergency"

class RunwayStatus(str, Enum):
    CLEAR = "clear"
    OCCUPIED = "occupied"
    CLOSED = "closed"

class WeatherCondition(str, Enum):
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    STORM = "storm"
    FOG = "fog"

class ActionType(str, Enum):
    ASSIGN_RUNWAY = "assign_runway"
    ASSIGN_GATE = "assign_gate"
    DELAY_FLIGHT = "delay_flight"
    DIVERT_FLIGHT = "divert_flight"
    CLEAR_RUNWAY = "clear_runway"
    DECLARE_EMERGENCY = "declare_emergency"


# ─── Sub-models ──────────────────────────────────────────────────────────────

class Flight(BaseModel):
    flight_id: str
    airline: str
    origin: str
    destination: str
    weight_class: WeightClass
    status: FlightStatus
    eta_minutes: float          # minutes from episode start
    fuel_remaining_min: float   # minutes of fuel remaining
    is_international: bool
    assigned_runway: Optional[str] = None
    assigned_gate: Optional[str] = None
    actual_landing_time: Optional[float] = None
    delay_minutes: float = 0.0
    passenger_count: int = 150

class Runway(BaseModel):
    runway_id: str
    name: str
    status: RunwayStatus
    current_flight: Optional[str] = None
    available_at: float = 0.0   # time when runway becomes free
    length_m: int = 3500

class Gate(BaseModel):
    gate_id: str
    terminal: str
    is_international: bool
    occupied: bool = False
    current_flight: Optional[str] = None
    available_at: float = 0.0

class Weather(BaseModel):
    condition: WeatherCondition
    visibility_km: float
    wind_speed_kts: float
    wind_direction_deg: int
    active_notams: List[str] = Field(default_factory=list)


# ─── OpenEnv typed models ─────────────────────────────────────────────────────

class AirportObservation(BaseModel):
    clock: float                          # simulated minutes elapsed
    flights: List[Flight]
    runways: List[Runway]
    gates: List[Gate]
    weather: Weather
    recent_events: List[str] = Field(default_factory=list)
    score_so_far: float = 0.0
    task_id: str = ""
    episode_done: bool = False

class AirportAction(BaseModel):
    action_type: ActionType
    flight_id: str
    runway_id: Optional[str] = None
    gate_id: Optional[str] = None
    delay_minutes: Optional[float] = None
    reason: Optional[str] = None

class AirportReward(BaseModel):
    value: float
    breakdown: Dict[str, float] = Field(default_factory=dict)
    message: str = ""


# ─── Separation rules ─────────────────────────────────────────────────────────

SEPARATION_MINUTES: Dict[Tuple[WeightClass, WeightClass], float] = {
    (WeightClass.SUPER,  WeightClass.SUPER):  3.0,
    (WeightClass.SUPER,  WeightClass.HEAVY):  4.0,
    (WeightClass.SUPER,  WeightClass.MEDIUM): 5.0,
    (WeightClass.SUPER,  WeightClass.LIGHT):  6.0,
    (WeightClass.HEAVY,  WeightClass.SUPER):  2.0,
    (WeightClass.HEAVY,  WeightClass.HEAVY):  3.0,
    (WeightClass.HEAVY,  WeightClass.MEDIUM): 4.0,
    (WeightClass.HEAVY,  WeightClass.LIGHT):  5.0,
    (WeightClass.MEDIUM, WeightClass.SUPER):  2.0,
    (WeightClass.MEDIUM, WeightClass.HEAVY):  2.0,
    (WeightClass.MEDIUM, WeightClass.MEDIUM): 2.0,
    (WeightClass.MEDIUM, WeightClass.LIGHT):  3.0,
    (WeightClass.LIGHT,  WeightClass.SUPER):  2.0,
    (WeightClass.LIGHT,  WeightClass.HEAVY):  2.0,
    (WeightClass.LIGHT,  WeightClass.MEDIUM): 2.0,
    (WeightClass.LIGHT,  WeightClass.LIGHT):  2.0,
}

def required_separation(prev: WeightClass, curr: WeightClass) -> float:
    return SEPARATION_MINUTES.get((prev, curr), 2.0)


# ─── Flight generators ────────────────────────────────────────────────────────

AIRLINES = ["AA", "UA", "DL", "SW", "BA", "LH", "AF", "EK", "QR", "SQ"]
AIRPORTS = ["JFK", "LAX", "ORD", "ATL", "DFW", "LHR", "CDG", "DXB", "SIN", "NRT"]

def _random_flight(flight_num: int, clock: float, spread_min: float = 30.0,
                   rng: Optional[random.Random] = None) -> Flight:
    r = rng or random
    airline = r.choice(AIRLINES)
    origin = r.choice(AIRPORTS)
    wc = r.choices(
        [WeightClass.LIGHT, WeightClass.MEDIUM, WeightClass.HEAVY, WeightClass.SUPER],
        weights=[10, 40, 40, 10]
    )[0]
    eta = clock + r.uniform(1, spread_min)
    fuel = eta + r.uniform(20, 60)   # fuel lasts beyond ETA
    intl = r.random() < 0.4
    pax = {"light": 50, "medium": 150, "heavy": 280, "super": 450}[wc.value]
    return Flight(
        flight_id=f"{airline}{100 + flight_num}",
        airline=airline,
        origin=origin,
        destination="HUB",
        weight_class=wc,
        status=FlightStatus.SCHEDULED,
        eta_minutes=eta,
        fuel_remaining_min=fuel,
        is_international=intl,
        passenger_count=pax + r.randint(-20, 20),
    )


# ─── Core Environment ─────────────────────────────────────────────────────────

class AirportEnv:
    TASK_IDS = ["single_runway_landing", "multi_runway_gate_assignment", "storm_disruption_recovery"]

    def __init__(self, task_id: str = "single_runway_landing", seed: int = 42):
        self.task_id = task_id
        self.seed = seed
        self._rng = random.Random(seed)
        self._state: Optional[AirportObservation] = None
        self._cumulative_reward = 0.0
        self._prev_runway_occupant: Dict[str, Optional[WeightClass]] = {}
        self._conflict_count = 0
        self._safe_landings = 0
        self._diversions = 0
        self._avoidable_diversions = 0
        self._emergency_resolved = 0

    # ── reset ──────────────────────────────────────────────────────────────

    def reset(self) -> AirportObservation:
        self._rng = random.Random(self.seed)
        self._cumulative_reward = 0.0
        self._prev_runway_occupant = {}
        self._conflict_count = 0
        self._safe_landings = 0
        self._diversions = 0
        self._avoidable_diversions = 0
        self._emergency_resolved = 0

        if self.task_id == "single_runway_landing":
            flights = [_random_flight(i, 0, 20, self._rng) for i in range(5)]
            runways = [Runway(runway_id="R1", name="09L/27R", status=RunwayStatus.CLEAR)]
            gates = [Gate(gate_id=f"G{i}", terminal="T1", is_international=False)
                     for i in range(1, 6)]
            weather = Weather(condition=WeatherCondition.CLEAR,
                              visibility_km=10.0, wind_speed_kts=5,
                              wind_direction_deg=90)

        elif self.task_id == "multi_runway_gate_assignment":
            flights = [_random_flight(i, 0, 30, self._rng) for i in range(10)]
            runways = [
                Runway(runway_id="R1", name="09L/27R", status=RunwayStatus.CLEAR),
                Runway(runway_id="R2", name="09R/27L", status=RunwayStatus.CLEAR),
            ]
            gates = (
                [Gate(gate_id=f"D{i}", terminal="T1", is_international=False) for i in range(1, 5)] +
                [Gate(gate_id=f"I{i}", terminal="T2", is_international=True)  for i in range(1, 5)]
            )
            weather = Weather(condition=WeatherCondition.CLOUDY,
                              visibility_km=8.0, wind_speed_kts=12,
                              wind_direction_deg=270)

        else:  # storm_disruption_recovery
            flights = [_random_flight(i, 0, 40, self._rng) for i in range(15)]
            # Give 4 flights low fuel → potential emergencies
            for f in self._rng.sample(flights, 4):
                f.fuel_remaining_min = f.eta_minutes + self._rng.uniform(5, 15)
            runways = [
                Runway(runway_id="R1", name="09L/27R", status=RunwayStatus.CLEAR),
                Runway(runway_id="R2", name="09R/27L", status=RunwayStatus.CLEAR),
            ]
            gates = (
                [Gate(gate_id=f"D{i}", terminal="T1", is_international=False) for i in range(1, 7)] +
                [Gate(gate_id=f"I{i}", terminal="T2", is_international=True)  for i in range(1, 5)]
            )
            weather = Weather(condition=WeatherCondition.RAIN,
                              visibility_km=5.0, wind_speed_kts=25,
                              wind_direction_deg=180,
                              active_notams=["Wind shear reported on approach RWY09R"])

        self._state = AirportObservation(
            clock=0.0,
            flights=flights,
            runways=runways,
            gates=gates,
            weather=weather,
            recent_events=["Episode started. Ready for ATC commands."],
            task_id=self.task_id,
        )
        for r in runways:
            self._prev_runway_occupant[r.runway_id] = None
        return self._state

    # ── state ──────────────────────────────────────────────────────────────

    def state(self) -> AirportObservation:
        if self._state is None:
            return self.reset()
        return self._state

    # ── step ───────────────────────────────────────────────────────────────

    def step(self, action: AirportAction) -> Tuple[AirportObservation, AirportReward, bool, Dict]:
        if self._state is None:
            self.reset()

        obs = self._state
        reward_breakdown: Dict[str, float] = {}
        events: List[str] = []
        total_reward = 0.0

        # Advance clock by 1 simulated minute per step
        obs.clock += 1.0

        # Storm trigger at t=20 for hard task
        if self.task_id == "storm_disruption_recovery" and abs(obs.clock - 20.0) < 0.5:
            r2 = self._get_runway("R2")
            if r2 and r2.status != RunwayStatus.CLOSED:
                r2.status = RunwayStatus.CLOSED
                obs.weather.condition = WeatherCondition.STORM
                obs.weather.visibility_km = 1.5
                obs.weather.wind_speed_kts = 55
                obs.weather.active_notams.append("RUNWAY 09R/27L CLOSED — storm damage")
                events.append("⚡ STORM: Runway R2 is now CLOSED. Divert or reassign affected flights.")

        # Passive: accumulate delay penalty for unresolved flights
        unresolved = [f for f in obs.flights
                      if f.status in (FlightStatus.SCHEDULED, FlightStatus.HOLDING)
                      and f.eta_minutes <= obs.clock]
        delay_penalty = -0.1 * len(unresolved)
        reward_breakdown["delay_penalty"] = delay_penalty
        total_reward += delay_penalty

        # Passive: check fuel emergencies
        for f in obs.flights:
            if f.status in (FlightStatus.SCHEDULED, FlightStatus.HOLDING):
                if f.fuel_remaining_min <= obs.clock + 5 and f.status != FlightStatus.EMERGENCY:
                    f.status = FlightStatus.EMERGENCY
                    events.append(f"🚨 EMERGENCY: {f.flight_id} has <5 min fuel! Priority landing required.")

        # ── Process action ──────────────────────────────────────────────────
        flight = self._get_flight(action.flight_id)
        if flight is None:
            events.append(f"⚠ Unknown flight {action.flight_id}")
            reward_breakdown["invalid_action"] = -0.5
            total_reward -= 0.5
        else:
            if action.action_type == ActionType.ASSIGN_RUNWAY:
                r, e = self._do_assign_runway(flight, action, obs)
                total_reward += r
                reward_breakdown["assign_runway"] = r
                events.extend(e)

            elif action.action_type == ActionType.ASSIGN_GATE:
                r, e = self._do_assign_gate(flight, action, obs)
                total_reward += r
                reward_breakdown["assign_gate"] = r
                events.extend(e)

            elif action.action_type == ActionType.DELAY_FLIGHT:
                mins = action.delay_minutes or 5.0
                flight.eta_minutes += mins
                flight.delay_minutes += mins
                r = -0.2 * mins
                total_reward += r
                reward_breakdown["delay"] = r
                events.append(f"⏱ {flight.flight_id} delayed {mins:.0f} min (new ETA {flight.eta_minutes:.0f})")

            elif action.action_type == ActionType.DIVERT_FLIGHT:
                could_have_landed = self._could_have_landed(flight, obs)
                flight.status = FlightStatus.DIVERTED
                self._diversions += 1
                if could_have_landed:
                    self._avoidable_diversions += 1
                    r = -2.0
                    events.append(f"✈ {flight.flight_id} diverted (AVOIDABLE — runway was available)")
                else:
                    r = -0.5
                    events.append(f"✈ {flight.flight_id} diverted to alternate airport")
                total_reward += r
                reward_breakdown["divert"] = r

            elif action.action_type == ActionType.CLEAR_RUNWAY:
                runway = self._get_runway(action.runway_id or "")
                if runway:
                    runway.status = RunwayStatus.CLEAR
                    runway.current_flight = None
                    events.append(f"✅ Runway {runway.name} cleared")
                    r = 0.1
                    total_reward += r
                    reward_breakdown["clear_runway"] = r

            elif action.action_type == ActionType.DECLARE_EMERGENCY:
                if flight.status == FlightStatus.EMERGENCY:
                    self._emergency_resolved += 1
                    r = 3.0
                    total_reward += r
                    reward_breakdown["emergency_resolved"] = r
                    events.append(f"🟢 Emergency cleared for {flight.flight_id} — priority landing granted")
                else:
                    r = -0.2
                    total_reward += r
                    reward_breakdown["false_emergency"] = r
                    events.append(f"⚠ {flight.flight_id} is not in emergency status")

        # Episode done check
        active = [f for f in obs.flights
                  if f.status not in (FlightStatus.LANDED, FlightStatus.DIVERTED,
                                      FlightStatus.DEPARTED)]
        done = len(active) == 0 or obs.clock >= 90

        if done:
            bonus = self._episode_bonus(obs)
            total_reward += bonus
            reward_breakdown["episode_bonus"] = bonus
            obs.episode_done = True
            events.append(f"🏁 Episode complete. Bonus: {bonus:+.2f}")

        self._cumulative_reward += total_reward
        obs.score_so_far = self._cumulative_reward
        obs.recent_events = events[-6:]

        reward = AirportReward(
            value=total_reward,
            breakdown=reward_breakdown,
            message=" | ".join(events[:2]) if events else "Step processed"
        )
        return obs, reward, done, {"cumulative_reward": self._cumulative_reward}

    # ── helpers ────────────────────────────────────────────────────────────

    def _get_flight(self, fid: str) -> Optional[Flight]:
        for f in self._state.flights:
            if f.flight_id == fid:
                return f
        return None

    def _get_runway(self, rid: str) -> Optional[Runway]:
        for r in self._state.runways:
            if r.runway_id == rid:
                return r
        return None

    def _get_gate(self, gid: str) -> Optional[Gate]:
        for g in self._state.gates:
            if g.gate_id == gid:
                return g
        return None

    def _do_assign_runway(self, flight: Flight, action: AirportAction,
                          obs: AirportObservation) -> Tuple[float, List[str]]:
        events = []
        runway = self._get_runway(action.runway_id or "")
        if runway is None:
            return -1.0, [f"❌ Runway {action.runway_id} not found"]
        if runway.status == RunwayStatus.CLOSED:
            return -2.0, [f"❌ Cannot assign {flight.flight_id} to CLOSED runway {runway.name}"]
        if runway.status == RunwayStatus.OCCUPIED:
            # Conflict!
            self._conflict_count += 1
            return -5.0, [f"💥 CONFLICT: {flight.flight_id} assigned to occupied runway {runway.name}!"]

        # Separation check
        prev_wc = self._prev_runway_occupant.get(runway.runway_id)
        sep_needed = required_separation(prev_wc, flight.weight_class) if prev_wc else 0
        gap = obs.clock - runway.available_at
        if prev_wc and gap < sep_needed:
            penalty = -1.0 * (sep_needed - gap)
            events.append(f"⚠ Separation violation on {runway.name}: needed {sep_needed:.0f}m, got {gap:.1f}m")
            self._conflict_count += 1
            return penalty, events

        # Successful assignment
        flight.assigned_runway = runway.runway_id
        flight.status = FlightStatus.LANDING
        flight.actual_landing_time = obs.clock
        runway.status = RunwayStatus.OCCUPIED
        runway.current_flight = flight.flight_id
        runway.available_at = obs.clock + 2.0   # 2 min runway occupancy
        self._prev_runway_occupant[runway.runway_id] = flight.weight_class
        self._safe_landings += 1
        delay = max(0, obs.clock - flight.eta_minutes)
        flight.delay_minutes += delay
        on_time_bonus = 0.3 if delay < 2 else 0.0
        events.append(f"✅ {flight.flight_id} cleared to land on {runway.name} (delay: {delay:.1f}m)")
        return 1.0 + on_time_bonus, events

    def _do_assign_gate(self, flight: Flight, action: AirportAction,
                        obs: AirportObservation) -> Tuple[float, List[str]]:
        gate = self._get_gate(action.gate_id or "")
        if gate is None:
            return -1.0, [f"❌ Gate {action.gate_id} not found"]
        if gate.occupied:
            self._conflict_count += 1
            return -5.0, [f"💥 CONFLICT: Gate {gate.gate_id} already occupied!"]
        if flight.is_international != gate.is_international:
            return -1.5, [f"❌ Gate type mismatch: {'intl' if flight.is_international else 'dom'} flight to {'dom' if flight.is_international else 'intl'} gate"]
        if flight.status not in (FlightStatus.LANDING, FlightStatus.LANDED, FlightStatus.SCHEDULED):
            return -0.5, [f"⚠ {flight.flight_id} not in a state that accepts gate assignment"]

        gate.occupied = True
        gate.current_flight = flight.flight_id
        gate.available_at = obs.clock + 45   # 45-min turnaround
        flight.assigned_gate = gate.gate_id
        flight.status = FlightStatus.LANDED
        timing_bonus = 0.5 if flight.actual_landing_time and (obs.clock - flight.actual_landing_time) < 5 else 0.0
        return 0.5 + timing_bonus, [f"🅿 {flight.flight_id} → Gate {gate.gate_id} (T{gate.terminal})"]

    def _could_have_landed(self, flight: Flight, obs: AirportObservation) -> bool:
        return any(r.status == RunwayStatus.CLEAR for r in obs.runways)

    def _episode_bonus(self, obs: AirportObservation) -> float:
        total = len(obs.flights)
        landed = sum(1 for f in obs.flights if f.status == FlightStatus.LANDED)
        diverted = sum(1 for f in obs.flights if f.status == FlightStatus.DIVERTED)
        on_time = sum(1 for f in obs.flights if f.status == FlightStatus.LANDED and f.delay_minutes < 5)
        conflicts = self._conflict_count

        throughput = landed / max(total, 1)
        on_time_ratio = on_time / max(landed, 1)
        conflict_penalty = min(conflicts * 1.0, 5.0)
        bonus = 10.0 * throughput * on_time_ratio - conflict_penalty
        return round(max(bonus, 0.0), 2)

    def grade(self) -> float:
        """Return a 0.0–1.0 score for the completed episode."""
        if self._state is None:
            return 0.0
        obs = self._state
        total = len(obs.flights)
        landed = sum(1 for f in obs.flights if f.status == FlightStatus.LANDED)
        avg_delay = (sum(f.delay_minutes for f in obs.flights if f.status == FlightStatus.LANDED)
                     / max(landed, 1))
        conflict_ratio = self._conflict_count / max(total, 1)
        emergency_ratio = self._emergency_resolved / max(
            sum(1 for f in obs.flights if f.status == FlightStatus.EMERGENCY) + self._emergency_resolved, 1)

        score = (
            0.40 * (landed / max(total, 1)) +
            0.25 * max(0, 1 - avg_delay / 20.0) +
            0.20 * (1 - conflict_ratio) +
            0.15 * emergency_ratio
        )
        return round(min(max(score, 0.0), 1.0), 4)
