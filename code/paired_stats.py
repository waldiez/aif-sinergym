#!/usr/bin/env python3
"""
paired_stats.py -- paired, per-seed comparison of controllers on W1 (noisy years).

Every controller saw the SAME weather for a given seed (the runner's weather
check verifies this), so differences are paired by seed.

  python paired_stats.py                                   # default comparisons
  python paired_stats.py --a frz:04_full --b rbc:proactive rbc:reactive rbc:dl
  python paired_stats.py --a frz:alt_fixedref --b rbc:pro_cm2p5

Run names: frz:<AIF W0 run>, onl:<AIF W0 run>, rbc:<rbc run without 'rbc_'>.
Lower is better for peak / CF / monthly peaks / costs; higher for CS / ZCR.
"""
import argparse, json, os

METRICS = [("peak_kW", "annual peak kW", -1), ("sum_monthly_peaks_kW", "sum monthly peaks kW", -1),
           ("dem_cost", "demand cost $", -1), ("en_cost", "energy cost $", -1),
           ("tot_cost", "total cost $", -1), ("CF", "coincidence factor", -1),
           ("CS", "comfort score", +1), ("CS_occ", "comfort score (occ)", +1),
           ("ZCR", "ZCR %", +1), ("energy_kWh", "energy kWh", -1)]


def label(spec, seed):
    mode, src = spec.split(":", 1)
    return f"tr_{mode}_{src}_var_s{seed}"


def load(root, lab):
    p = os.path.join(root, lab, "results.json")
    if not os.path.exists(p):
        return None
    eps = json.load(open(p)).get("episodes", [])
    return eps[0] if eps else None


def compare(root, a, b, seeds):
    ra = {s: load(root, label(a, s)) for s in seeds}
    rb = {s: load(root, label(b, s)) for s in seeds}
    ok = [s for s in seeds if ra[s] and rb[s]]
    if not ok:
        print(f"\n{a} vs {b}: no paired runs found"); return
    print(f"\n{a}  minus  {b}   (seeds {ok})")
    print(f"  {'metric':24s}" + "".join(f"{'s'+str(s):>9s}" for s in ok)
          + f"{'mean':>10s}{'sd':>8s}  A better")
    for key, name, sign in METRICS:
        d = [ra[s].get(key) - rb[s].get(key) for s in ok
             if ra[s].get(key) is not None and rb[s].get(key) is not None]
        if len(d) != len(ok):
            continue
        mu = sum(d) / len(d)
        sd = (sum((x - mu) ** 2 for x in d) / (len(d) - 1)) ** 0.5 if len(d) > 1 else 0.0
        wins = sum(1 for x in d if sign * x > 0)
        fmt = ".3f" if abs(mu) < 5 and key in ("CF", "CS", "CS_occ") else ".1f" \
            if key in ("peak_kW", "sum_monthly_peaks_kW", "ZCR") else ".0f"
        print(f"  {name:24s}" + "".join(f"{format(x, '+' + fmt):>9s}" for x in d)
              + f"{format(mu, '+' + fmt):>10s}{format(sd, fmt):>8s}  {wins}/{len(d)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ab_runs")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--a", default="frz:04_full")
    ap.add_argument("--b", nargs="+",
                    default=["rbc:proactive", "rbc:reactive", "rbc:dl", "frz:01_baseline"])
    x = ap.parse_args()
    seeds = [int(s) for s in x.seeds.split(",")]
    for b in x.b:
        compare(x.root, x.a, b, seeds)


if __name__ == "__main__":
    main()
