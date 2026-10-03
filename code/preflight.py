#!/usr/bin/env python3
"""
Preflight checks on the REAL Sinergym environment (run once before the sweep).

  python preflight.py              # checks 1-3 (~3-5 min: one EnergyPlus year)
  python preflight.py --legacy     # + check 4: v15 --legacy_infer vs v14 (~10 min)

 1. Observation layout: obs[9:24] are the 15 occupied zones (not plenums) and
    obs[92:95] are the three per-floor meters.
 2. Calendar alignment: the metrics' weekday 07-19 'occupied' flag vs the
    EnergyPlus occupancy signal obs[45:60] over a full year.
 3. Weather variability: same seed -> same weather, different seed -> different.
 4. (optional) v15 in legacy mode reproduces v14's first-episode summary.
"""
import argparse, os, subprocess, sys
from datetime import date
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from register_env import make_custom_env, OCCUPIED_ZONES  # noqa: E402

OK, BAD = "  [ OK ]", "  [FAIL]"


def names(env):
    for attr in ("observation_variables",):
        for obj in (env, getattr(env, "unwrapped", env)):
            v = getattr(obj, attr, None)
            if v is not None:
                return list(v)
    try:
        return list(env.get_wrapper_attr("observation_variables"))
    except Exception:
        return None


def check_layout():
    print("1) observation layout")
    env = make_custom_env(weather="mixed")
    n = names(env)
    if n is None:
        print("  could not read observation variable names; print env.observation_variables")
        env.close(); return False
    zt = n[9:24]
    exp = [f"air_temperature_{z}" for z in OCCUPIED_ZONES]
    ok1 = zt == exp
    print(f"{OK if ok1 else BAD} obs[9:24] -> {zt[0]} ... {zt[-1]}")
    if not ok1:
        for i, (a, b) in enumerate(zip(zt, exp)):
            if a != b: print(f"        obs[{9+i}] = {a}   expected {b}")
    print(f"        obs[24:27] = {n[24:27]}")
    fp = n[92:95] if len(n) >= 95 else None
    ok2 = fp is not None and all("floor_power" in x for x in fp)
    print(f"{OK if ok2 else BAD} obs[92:95] = {fp}")
    print(f"        obs[90], obs[91] = {n[90]}, {n[91]}   (len {len(n)})")
    env.close()
    return ok1 and ok2


def check_calendar():
    print("2) calendar alignment over one year (constant 21/24 setpoints)")
    env = make_custom_env(weather="mixed")
    obs, _ = env.reset(seed=0)
    act = np.array([21.0] * 15 + [24.5] * 15, dtype=np.float32)
    n = agree = metric_occ_empty = eplus_occ_outside = 0
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(act); done = term or trunc
        m, d, h = int(obs[0]), int(obs[1]), int(obs[2])
        try: wk = date(2024, m, d).weekday() >= 5
        except ValueError: wk = False
        metric_occ = (not wk) and 7 <= h <= 19
        eplus_occ = bool(np.any(obs[45:60] > 0))
        n += 1; agree += metric_occ == eplus_occ
        metric_occ_empty += metric_occ and not eplus_occ
        eplus_occ_outside += eplus_occ and not metric_occ
    env.close()
    frac_bad = metric_occ_empty / max(n, 1)
    print(f"        agreement {100*agree/n:.1f}% of {n} steps")
    print(f"{OK if frac_bad < 0.01 else '  [WARN]'} metric says occupied but EnergyPlus has nobody: "
          f"{metric_occ_empty} steps ({100*frac_bad:.2f}%)")
    print(f"        EnergyPlus occupied outside the metric window: {eplus_occ_outside} steps "
          f"(early arrivals / late leavers / Saturdays are expected here)")
    if frac_bad >= 0.01:
        print("        -> inspect which dates: holidays, or a weekday offset between the "
              "2024 calendar used by the metrics and the EnergyPlus RunPeriod")
    return True   # informative, not blocking


def first_temps(sigma, seed, k=300):
    np.random.seed(seed)
    env = make_custom_env(weather="mixed", weather_variability=sigma)
    obs, _ = env.reset(seed=seed)
    out = [float(obs[3])]
    a = np.array([21.0] * 15 + [24.5] * 15, dtype=np.float32)
    for _ in range(k):
        obs, *_ = env.step(a); out.append(float(obs[3]))
    env.close()
    return np.array(out)


def check_variability():
    print("3) weather variability")
    a1, a1b, a2 = first_temps(1.5, 1), first_temps(1.5, 1), first_temps(1.5, 2)
    base = first_temps(0.0, 1)
    same = np.allclose(a1, a1b); diff = not np.allclose(a1, a2)
    noisy = not np.allclose(a1, base)
    print(f"{OK if same else BAD} same seed reproduces the weather")
    print(f"{OK if diff else BAD} different seeds give different weather "
          f"(mean |dT| {np.mean(np.abs(a1 - a2)):.2f} C)")
    print(f"{OK if noisy else BAD} sigma>0 differs from the deterministic TMY3 year")
    return same and diff and noisy


def check_legacy():
    print("4) v15 --legacy_infer --horizon_ref 0 vs v14 (1 episode each)")
    common = ["--mode", "zone", "--structural", "--weather", "mixed", "--episodes", "1",
              "--policy_len", "8", "--energy_weight", ".2", "--deadband_weight", "8.0",
              "--floor_power_idx", "92,93,94", "--floor_power_scale", "0.0011111",
              "--congestion_weight", "5", "--action_temp", "0", "--tou_weight", "0",
              "--learn_C", "--c_lr", "0.02", "--energy_w_min", "0", "--energy_w_max", "1.0",
              "--no-save"]
    outs = {}
    for tag, script, extra in (("v14", os.path.join("tests", "aif_agent_v14_reference.py"), []),
                               ("v15", "aif_agent.py",
                                ["--legacy_infer", "--horizon_ref", "0",
                                 "--out_dir", "preflight_v15"])):
        p = subprocess.run([sys.executable, os.path.join(HERE, script)] + common + extra,
                           capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": HERE})
        keep = [l for l in p.stdout.splitlines()
                if any(k in l for k in ("Peak Demand", "Coincidence Factor", "Total Cost",
                                        "CS (weighted", "Zone-Comfort"))]
        outs[tag] = keep
        print(f"        {tag}: " + " | ".join(x.strip() for x in keep))
    ok = outs["v14"] == outs["v15"] and outs["v14"]
    print(f"{OK if ok else BAD} identical episode summaries")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--legacy", action="store_true")
    a = ap.parse_args()
    res = [check_layout(), check_calendar(), check_variability()]
    if a.legacy:
        res.append(check_legacy())
    print("\nPREFLIGHT", "PASSED" if all(res) else "HAS FAILURES (see above)")
