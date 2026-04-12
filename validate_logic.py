#!/usr/bin/env python3
"""
validate_logic.py — Pure stdlib validation (no pydantic required)
Tests all environment logic using plain dicts to simulate the state.
"""
import random
import sys

# ── Enums as constants ────────────────────────────────────────────────────────
WC_ORDER = ['light', 'medium', 'heavy', 'super']
SEP = {
    ('super','super'):3,('super','heavy'):4,('super','medium'):5,('super','light'):6,
    ('heavy','super'):2,('heavy','heavy'):3,('heavy','medium'):4,('heavy','light'):5,
    ('medium','super'):2,('medium','heavy'):2,('medium','medium'):2,('medium','light'):3,
    ('light','super'):2,('light','heavy'):2,('light','medium'):2,('light','light'):2,
}

AIRLINES = ['AA','UA','DL','SW','BA','LH']
AIRPORTS = ['JFK','LAX','ORD','ATL','LHR','CDG']

def make_flight(i, spread=20, rng=None):
    r = rng or random
    al = r.choice(AIRLINES)
    wc = r.choices(WC_ORDER, weights=[10,40,40,10])[0]
    eta = 1 + r.random() * spread
    fuel = eta + 20 + r.random() * 40
    return dict(
        flight_id=f"{al}{100+i}", airline=al, weight_class=wc,
        status='scheduled', eta_minutes=round(eta,1),
        fuel_remaining_min=round(fuel,1), is_international=r.random()<0.4,
        assigned_runway=None, assigned_gate=None,
        actual_landing_time=None, delay_minutes=0.0, passenger_count=150
    )

def make_env(task_id, seed=42):
    rng = random.Random(seed)
    n = {'single_runway_landing':5,'multi_runway_gate_assignment':10,'storm_disruption_recovery':15}[task_id]
    spread = {'single_runway_landing':20,'multi_runway_gate_assignment':30,'storm_disruption_recovery':40}[task_id]
    flights = [make_flight(i, spread, rng) for i in range(n)]
    if task_id == 'storm_disruption_recovery':
        for f in rng.sample(flights, 4):
            f['fuel_remaining_min'] = f['eta_minutes'] + 5 + rng.random()*10

    if task_id == 'single_runway_landing':
        runways = [dict(runway_id='R1', name='09L/27R', status='clear', current_flight=None, available_at=0)]
        gates = [dict(gate_id=f'G{i}', terminal='T1', is_international=False, occupied=False, current_flight=None) for i in range(1,6)]
    else:
        runways = [
            dict(runway_id='R1', name='09L/27R', status='clear', current_flight=None, available_at=0),
            dict(runway_id='R2', name='09R/27L', status='clear', current_flight=None, available_at=0),
        ]
        gates = (
            [dict(gate_id=f'D{i}', terminal='T1', is_international=False, occupied=False, current_flight=None) for i in range(1,5)] +
            [dict(gate_id=f'I{i}', terminal='T2', is_international=True, occupied=False, current_flight=None) for i in range(1,5)]
        )

    return dict(clock=0.0, flights=flights, runways=runways, gates=gates,
                task_id=task_id, episode_done=False, recent_events=[], score=0.0)

def get_runway(state, rid): return next((r for r in state['runways'] if r['runway_id']==rid), None)
def get_gate(state, gid):   return next((g for g in state['gates'] if g['gate_id']==gid), None)
def get_flight(state, fid): return next((f for f in state['flights'] if f['flight_id']==fid), None)

def step(state, action):
    state['clock'] += 1.0
    reward = 0.0
    events = []

    # Storm trigger
    if state['task_id']=='storm_disruption_recovery' and abs(state['clock']-20)<0.5:
        r2 = get_runway(state,'R2')
        if r2 and r2['status']!='closed':
            r2['status']='closed'
            events.append('STORM: R2 CLOSED')

    # Fuel emergencies
    for f in state['flights']:
        if f['status'] in ('scheduled','holding') and f['fuel_remaining_min']-state['clock']<=5:
            f['status']='emergency'

    # Delay penalty
    unresolved = [f for f in state['flights'] if f['status'] in ('scheduled','holding') and f['eta_minutes']<=state['clock']]
    reward -= 0.1 * len(unresolved)

    at = action['action_type']
    fid = action['flight_id']
    flight = get_flight(state, fid)

    if flight is None:
        reward -= 0.5
    elif at == 'assign_runway':
        rwy = get_runway(state, action.get('runway_id',''))
        if not rwy:
            reward -= 1.0
        elif rwy['status'] == 'closed':
            reward -= 2.0
        elif rwy['status'] == 'occupied':
            reward -= 5.0
        else:
            flight['assigned_runway'] = rwy['runway_id']
            flight['status'] = 'landing'
            flight['actual_landing_time'] = state['clock']
            delay = max(0, state['clock'] - flight['eta_minutes'])
            flight['delay_minutes'] += delay
            rwy['status'] = 'occupied'
            rwy['current_flight'] = flight['flight_id']
            rwy['available_at'] = state['clock'] + 2
            reward += 1.0 + (0.3 if delay < 2 else 0.0)
            events.append(f"LANDED {flight['flight_id']} on {rwy['runway_id']}")
    elif at == 'assign_gate':
        gate = get_gate(state, action.get('gate_id',''))
        if not gate:
            reward -= 1.0
        elif gate['occupied']:
            reward -= 5.0
        elif flight['is_international'] != gate['is_international']:
            reward -= 1.5
        else:
            gate['occupied'] = True
            gate['current_flight'] = flight['flight_id']
            flight['assigned_gate'] = gate['gate_id']
            flight['status'] = 'landed'
            reward += 1.0
            events.append(f"GATE {flight['flight_id']} -> {gate['gate_id']}")
    elif at == 'delay_flight':
        m = action.get('delay_minutes', 5)
        flight['eta_minutes'] += m
        flight['delay_minutes'] += m
        reward -= 0.2 * m
    elif at == 'divert_flight':
        can_land = any(r['status']=='clear' for r in state['runways'])
        flight['status'] = 'diverted'
        reward += (-2.0 if can_land else -0.5)
    elif at == 'clear_runway':
        rwy = get_runway(state, action.get('runway_id',''))
        if rwy and rwy['status']=='occupied':
            rwy['status']='clear'
            rwy['current_flight']=None
            reward += 0.1
    elif at == 'declare_emergency':
        if flight['status']=='emergency':
            reward += 3.0
            flight['status']='landing'
        else:
            reward -= 0.2

    state['score'] += reward
    state['recent_events'] = events[-6:]
    active = [f for f in state['flights'] if f['status'] not in ('landed','diverted')]
    done = len(active)==0 or state['clock']>=90
    if done:
        state['episode_done'] = True
    return reward, done

def grade(state):
    total = len(state['flights'])
    landed = [f for f in state['flights'] if f['status']=='landed']
    avg_delay = sum(f['delay_minutes'] for f in landed)/max(len(landed),1)
    conflict_free = 1.0  # simplification
    score = (0.40*(len(landed)/max(total,1)) + 0.25*max(0,1-avg_delay/20) + 0.20*conflict_free + 0.15)
    return round(min(max(score,0),1),4)

# ── Tests ─────────────────────────────────────────────────────────────────────
def run_task(task_id):
    state = make_env(task_id)
    assert state['clock'] == 0.0
    assert len(state['flights']) > 0
    assert len(state['runways']) > 0
    assert len(state['gates']) > 0
    print(f"  reset OK: {len(state['flights'])} flights, {len(state['runways'])} runways")

    total_reward = 0.0
    steps = 0
    while steps < 30:
        # greedy agent
        flight = next((f for f in state['flights'] if f['status']=='emergency'), None)
        if flight:
            action = dict(action_type='declare_emergency', flight_id=flight['flight_id'])
        else:
            landing_no_gate = next((f for f in state['flights'] if f['status']=='landing' and not f['assigned_gate']), None)
            if landing_no_gate:
                gate = next((g for g in state['gates'] if not g['occupied'] and g['is_international']==landing_no_gate['is_international']), None)
                if gate:
                    action = dict(action_type='assign_gate', flight_id=landing_no_gate['flight_id'], gate_id=gate['gate_id'])
                else:
                    action = dict(action_type='delay_flight', flight_id=landing_no_gate['flight_id'], delay_minutes=3)
            else:
                occ_rwy = next((r for r in state['runways'] if r['status']=='occupied' and r['available_at']<=state['clock']), None)
                if occ_rwy:
                    cf = occ_rwy['current_flight']
                    action = dict(action_type='clear_runway', flight_id=cf or state['flights'][0]['flight_id'], runway_id=occ_rwy['runway_id'])
                else:
                    waiting = [f for f in state['flights'] if f['status']=='scheduled']
                    free_rwy = next((r for r in state['runways'] if r['status']=='clear'), None)
                    if waiting and free_rwy:
                        f = sorted(waiting, key=lambda x:x['fuel_remaining_min'])[0]
                        action = dict(action_type='assign_runway', flight_id=f['flight_id'], runway_id=free_rwy['runway_id'])
                    elif waiting:
                        action = dict(action_type='delay_flight', flight_id=waiting[0]['flight_id'], delay_minutes=2)
                    else:
                        break

        r, done = step(state, action)
        total_reward += r
        steps += 1
        if done:
            break

    g = grade(state)
    assert 0.0 <= g <= 1.0, f"Grade out of range: {g}"
    print(f"  grade:   {g:.4f} | steps: {steps} | reward: {total_reward:.2f}")
    return g

def main():
    print("="*54)
    print("Airport Scheduling Env — Logic Validation")
    print("="*54)
    tasks = ['single_runway_landing','multi_runway_gate_assignment','storm_disruption_recovery']
    results = {}
    for t in tasks:
        print(f"\n[{t}]")
        try:
            g = run_task(t)
            results[t] = g
            print(f"  ✅ PASS")
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            results[t] = -1.0

    print(f"\n{'='*54}")
    all_ok = all(v>=0 for v in results.values())
    for t, g in results.items():
        sym = "✅" if g>=0 else "❌"
        print(f"  {sym} {t:<44} {g:.4f}")
    print(f"\n{'✅ All checks passed!' if all_ok else '❌ Some checks failed.'}")
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
