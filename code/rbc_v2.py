#!/usr/bin/env python3
"""
rbc_v2.py -- unified rule-based baselines with the SAME weather protocol,
scoring and results.json schema as aif_agent.py.

The reactive and proactive controllers are imported UNCHANGED from
rbc_office_reactive.py and proactive_rbc.py (same code base, two switches:
structural offsets and 04-06 pre-conditioning). This file only adds the runner
and one new baseline:

  --variant reactive    rbc_office_reactive.RuleBasedController
  --variant proactive   proactive_rbc.RuleBasedController (structural=True)
  --variant dl          proactive + two textbook demand-management layers:
      (1) staggered pre-conditioning: floor f starts pre-conditioning
          f * --stagger_min minutes after 04:00 (until then it keeps the
          reactive controller's setback);
      (2) demand limiting with hysteresis: when the metered building HVAC
          demand (obs[90], last step) exceeds --dl_limit_frac * --dl_ref_W,
          cooling setpoints go up and heating setpoints go down by --dl_shed_C
          ('global': all zones; 'rotate': one floor at a time, hourly
          rotation); released below --dl_release_frac * limit.
      It answers the reviewer question "would a simple rule not also stagger
      the floors and shave the peak?"

Scoring: metrics_utils (same functions as AIF/MADDPG); deadband violation =
htg >= clg - 2 over all zone-steps (same rule as AIF/MADDPG).
Weather: --seed / --weather_variability reproduce the AIF runs' noisy years
(same seeding order: np.random.seed(seed) -> make env -> reset(seed=seed)).

  python rbc_v2.py --variant proactive --out_dir rbc_runs/rbc_proactive
  python rbc_v2.py --variant dl --weather_variability 1.5 --seed 3 --out_dir ...
"""
import argparse, csv, hashlib, importlib, json, os, sys, time
from datetime import date
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IDX_MONTH, IDX_DAY, IDX_HOUR, IDX_OUT, IDX_DEMAND = 0, 1, 2, 3, 90
N_Z = 15


def zone_floor(names):
    fl = []
    for n in names:
        s = n.lower()
        fl.append(0 if "bot" in s else 1 if "mid" in s else 2 if "top" in s else -1)
    assert all(f >= 0 for f in fl), f"cannot parse floors from {names}"
    return np.array(fl)


def pre_occupied(month, day, hour):
    try:
        wk = date(2024, int(month), int(day)).weekday() >= 5
    except ValueError:
        wk = False
    return (not wk) and 4 <= hour < 6


class DemandLimitingRBC:
    """Proactive RBC + staggered pre-conditioning + demand limiting."""

    def __init__(self, pro, rea, floors, low, high, limit_W, shed_C=1.0,
                 release_frac=0.9, mode="global", stagger_min=40):
        self.pro, self.rea, self.floors = pro, rea, floors
        self.low, self.high = np.asarray(low), np.asarray(high)
        self.limit_W, self.shed_C = limit_W, shed_C
        self.release_W = release_frac * limit_W if limit_W else None
        self.mode, self.stagger_min = mode, stagger_min
        self.shedding = False
        self._hour, self._k = None, 0
        self.n_shed_steps = self.n_steps = self.n_stagger_zone_steps = 0

    def get_action(self, obs, info=None):
        a = np.asarray(self.pro.get_action(obs, info), dtype=np.float64).copy()
        m, d, h = int(obs[IDX_MONTH]), int(obs[IDX_DAY]), int(obs[IDX_HOUR])
        # minute within the hour from the 15-min step counter
        if h != self._hour:
            self._hour, self._k = h, 0
        else:
            self._k += 1
        minute_since_4 = (h - 4) * 60 + 15 * self._k
        self.n_steps += 1
        # (1) staggered pre-conditioning start by floor
        if self.stagger_min > 0 and pre_occupied(m, d, h):
            r = np.asarray(self.rea.get_action(obs, info), dtype=np.float64)
            late = self.floors * self.stagger_min > minute_since_4
            idx = np.where(late)[0]
            a[idx] = r[idx]; a[N_Z + idx] = r[N_Z + idx]
            self.n_stagger_zone_steps += len(idx)
        # (2) demand limiting with hysteresis on last step's metered demand
        if self.limit_W:
            dem = float(obs[IDX_DEMAND])
            if not self.shedding and dem > self.limit_W:
                self.shedding = True
            elif self.shedding and dem < self.release_W:
                self.shedding = False
            if self.shedding:
                self.n_shed_steps += 1
                zones = (np.arange(N_Z) if self.mode == "global"
                         else np.where(self.floors == (h % 3))[0])
                a[zones] -= self.shed_C
                a[N_Z + zones] += self.shed_C
        return np.clip(a, self.low, self.high).astype(np.float32)

    def stats(self):
        return {"dl_shed_frac": self.n_shed_steps / max(self.n_steps, 1),
                "dl_stagger_zone_steps": int(self.n_stagger_zone_steps)}


def run(args):
    from register_env import make_custom_env, OCCUPIED_ZONES
    from metrics_utils import (CustomRewardWrapper, CSAccumulator,
                               compute_hourly_linear_reward, compute_zone_comfort_rate,
                               compute_mean_deviation, load_factor,
                               floor_coincidence_factor, DEFAULT_TIMESTEP_HOURS)
    import metrics_utils as _mu
    os.makedirs(args.out_dir, exist_ok=True)
    P = lambda n: os.path.join(args.out_dir, n)
    fpi = [int(x) for x in args.floor_power_idx.split(",")] if args.floor_power_idx else None
    fps = args.floor_power_scale

    # same seeding order as the AIF runner -> identical noisy weather per seed
    np.random.seed(int(args.seed))
    kw = ({"weather_variability": float(args.weather_variability)}
          if args.weather_variability and args.weather_variability > 0 else {})
    env = CustomRewardWrapper(make_custom_env(weather=args.weather, real_world=False, **kw),
                              timestep_hours=DEFAULT_TIMESTEP_HOURS)

    pro_mod = importlib.import_module(args.proactive_module)
    rea_mod = importlib.import_module(args.reactive_module)
    if args.variant == "reactive":
        agent = rea_mod.RuleBasedController(env, granularity="zone", structural=False,
                                            cool_margin=args.cool_margin)
    elif args.variant == "proactive":
        agent = pro_mod.RuleBasedController(env, granularity="zone", structural=True,
                                            cool_margin=args.cool_margin)
    else:
        pro = pro_mod.RuleBasedController(env, granularity="zone", structural=True,
                                          cool_margin=args.cool_margin)
        rea = rea_mod.RuleBasedController(env, granularity="zone", structural=False,
                                          cool_margin=args.cool_margin)
        limit = args.dl_limit_frac * args.dl_ref_W if args.dl_limit_frac > 0 else None
        agent = DemandLimitingRBC(pro, rea, zone_floor(OCCUPIED_ZONES),
                                  env.action_space.low, env.action_space.high,
                                  limit, args.dl_shed_C, args.dl_release_frac,
                                  args.dl_mode, args.stagger_min)

    return run_episode(env, agent, args, "rbc_v2.py", f"RBC-{args.variant}")


def run_episode(env, agent, args, script, name):
    """Controller-agnostic evaluation episode: identical scoring, logs and
    results.json schema for every controller (RBC, MADDPG, ...). `agent` only
    needs get_action(obs, info) -> 30 setpoints; optional agent.stats()."""
    from register_env import OCCUPIED_ZONES
    from metrics_utils import (CSAccumulator, compute_hourly_linear_reward,
                               compute_zone_comfort_rate, compute_mean_deviation,
                               load_factor, floor_coincidence_factor)
    import metrics_utils as _mu
    os.makedirs(args.out_dir, exist_ok=True)
    P = lambda n: os.path.join(args.out_dir, n)
    fpi = [int(x) for x in args.floor_power_idx.split(",")] if args.floor_power_idx else None
    fps = args.floor_power_scale
    results = {"label": args.label, "script": script, "config": vars(args),
               "weather": args.weather, "seed": int(args.seed),
               "weather_variability": float(args.weather_variability or 0.0),
               "started_fresh": True, "loaded_from": None,
               "metrics_utils_md5": hashlib.md5(open(_mu.__file__, "rb").read()).hexdigest()[:10],
               "episodes": []}
    zf = open(P("zone_temps_actions.csv"), "w", newline=""); zw = csv.writer(zf)
    zw.writerow(["episode", "step", "month", "day", "hour", "outdoor_temp"]
                + [f"T_{z}" for z in OCCUPIED_ZONES] + [f"HTG_{z}" for z in OCCUPIED_ZONES]
                + [f"CLG_{z}" for z in OCCUPIED_ZONES])
    t0 = time.time()
    obs, info = env.reset(seed=int(args.seed))
    cs = CSAccumulator(); energy, floor_rows, outdoor = [], [], []
    zok = zc = dev = dc = 0.0; dbv = dbt = 0; cust = lin = hlr = 0.0
    m_peak, peak_ev = {}, {"kW": -1.0}; step = 0; done = False
    while not done:
        action = np.asarray(agent.get_action(obs, info), dtype=np.float64)
        step += 1
        zw.writerow([1, step, int(obs[IDX_MONTH]), int(obs[IDX_DAY]), int(obs[IDX_HOUR]),
                     f"{float(obs[IDX_OUT]):.3f}"]
                    + [f"{float(obs[9 + i]):.3f}" for i in range(N_Z)]
                    + [f"{action[i]:.3f}" for i in range(N_Z)]
                    + [f"{action[N_Z + i]:.3f}" for i in range(N_Z)])
        outdoor.append(round(float(obs[IDX_OUT]), 2))
        obs_pre = obs
        obs, reward, term, trunc, info = env.step(action.astype(np.float32))
        done = term or trunc
        cust += info.get("custom_reward", 0.0); lin += info.get("original_reward", 0.0)
        h_, _ = compute_hourly_linear_reward(obs); hlr += h_
        a_, b_ = compute_zone_comfort_rate(obs, occupied_only=True); zok += a_; zc += b_
        a_, b_ = compute_mean_deviation(obs, occupied_only=True); dev += a_; dc += b_
        cs.update(obs)
        dbt += N_Z; dbv += int(np.sum(action[:N_Z] >= action[N_Z:] - 2.0))
        pwr = float(obs[IDX_DEMAND]); energy.append(pwr)
        mo = int(obs[IDX_MONTH]); m_peak[mo] = max(m_peak.get(mo, 0.0), pwr / 1000.0)
        if fpi:
            floor_rows.append([float(obs[i]) * fps for i in fpi])
        if pwr / 1000.0 > peak_ev["kW"]:
            T = np.array([float(obs_pre[9 + i]) for i in range(N_Z)])
            h, c = action[:N_Z], action[N_Z:]
            peak_ev = {"kW": pwr / 1000.0, "step": step, "month": mo,
                       "day": int(obs[IDX_DAY]), "hour": int(obs[IDX_HOUR]),
                       "outdoor_C": float(obs_pre[IDX_OUT]),
                       "floor_kW": ([float(obs[i]) * fps / 1000.0 for i in fpi] if fpi else None),
                       "n_heating": int(np.sum(T < h - 0.1)), "n_cooling": int(np.sum(T > c + 0.1)),
                       "n_band_lt2": int(np.sum(c - h < 2.0))}
    zf.close(); env.close()

    rec = {"episode": 1, "peak_kW": max(energy) / 1000.0, "LF": float(load_factor(energy)),
           "energy_kWh": sum(energy) * 0.25 / 1000,
           "en_cost": float(info.get("cost_energy_usd", 0.0)),
           "dem_cost": float(info.get("cost_demand_usd", 0.0)),
           "tot_cost": float(info.get("cost_total_usd", info.get("cost_energy_usd", 0.0)
                                      + info.get("cost_demand_usd", 0.0))),
           "CS": float(cs.mean), "CS_occ": float(cs.mean_occ),
           "ZCR": zok / zc * 100 if zc else float("nan"), "mean_dev": dev / dc if dc else 0.0,
           "db_viol": dbv / dbt * 100, "custom_reward": cust, "linear_reward": lin,
           "hourly_reward": hlr,
           "monthly_peak_kW": {str(k): v for k, v in sorted(m_peak.items())},
           "sum_monthly_peaks_kW": float(sum(m_peak.values())), "peak_event": peak_ev,
           "override_rate": None,
           "weather_fp": hashlib.md5(json.dumps(outdoor).encode()).hexdigest()[:12],
           "wall_s": time.time() - t0}
    if fpi and floor_rows:
        fp = np.asarray(floor_rows)
        np.save(P("floorlog_ep1.npy"), fp)
        rec.update({"CF": float(floor_coincidence_factor(fp)),
                    "coincident_peak_kW": float(fp.sum(1).max() / 1000.0),
                    "floor_peaks_kW": [float(x) for x in fp.max(0) / 1000.0]})
    if hasattr(agent, "stats"):
        rec.update(agent.stats())
    results["episodes"].append(rec)
    with open(args.results_json or P("results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"{name} | {args.weather} seed={args.seed} sigma={args.weather_variability} | "
          f"peak {rec['peak_kW']:.1f} kW  CF {rec.get('CF', float('nan')):.3f}  "
          f"CS {rec['CS']:.3f}/{rec['CS_occ']:.3f}  ZCR {rec['ZCR']:.1f}%  "
          f"DB {rec['db_viol']:.1f}%  E {rec['energy_kWh']:,.0f} kWh  "
          f"${rec['tot_cost']:,.0f}  ({rec['wall_s']:.0f}s)")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--variant", choices=["reactive", "proactive", "dl"], default="proactive")
    ap.add_argument("--weather", default="mixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weather_variability", type=float, default=0.0)
    ap.add_argument("--cool_margin", type=float, default=1.0)
    ap.add_argument("--dl_ref_W", type=float, default=92400.0,
                    help="reference peak (W); default = RBC-proactive W0 annual peak")
    ap.add_argument("--dl_limit_frac", type=float, default=0.9,
                    help="demand limit as a fraction of dl_ref_W (0 disables limiting)")
    ap.add_argument("--dl_release_frac", type=float, default=0.9)
    ap.add_argument("--dl_shed_C", type=float, default=1.0)
    ap.add_argument("--dl_mode", choices=["global", "rotate"], default="global")
    ap.add_argument("--stagger_min", type=int, default=40,
                    help="pre-conditioning start offset per floor, minutes (0 disables)")
    ap.add_argument("--floor_power_idx", default="92,93,94")
    ap.add_argument("--floor_power_scale", type=float, default=0.0011111)
    ap.add_argument("--proactive_module", default="proactive_rbc")
    ap.add_argument("--reactive_module", default="rbc_office_reactive")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--results_json", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--episodes", type=int, default=1, help="ignored: RBCs are deterministic")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
