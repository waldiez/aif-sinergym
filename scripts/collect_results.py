#!/usr/bin/env python3
"""
Copy the small, publishable result files from ab_runs/ into results/:

  results/runs/<label>/results.json      every run (per-episode metrics)
  results/runs/<label>/run.log           the exact command line and console output
  results/runs/alt_fixedref/...          floor logs + energy weights (Fig. 2, weights)
  results/runs/rbc_proactive/...         floor log + zone log (Fig. 2)
  results/results.csv, robustness.csv    the summary tables
  results/maddpg/<name>_metadata.json    MADDPG training curves (no buffers)

make_figures.py can then be run with --root results/runs, so every table and
figure can be regenerated without re-running any simulation.
"""
import argparse, glob, os, shutil

FIGURE_FILES = {"alt_fixedref": ["floorlog_ep*.npy", "energy_w_daily_ep*.npy"],
                "rbc_proactive": ["floorlog_ep1.npy", "zone_temps_actions.csv"]}

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="ab_runs")
ap.add_argument("--dest", default="../results")
ap.add_argument("--maddpg_meta", nargs="*", default=[])
a = ap.parse_args()

n = 0
for res in sorted(glob.glob(os.path.join(a.root, "*", "results.json"))):
    label = os.path.basename(os.path.dirname(res))
    out = os.path.join(a.dest, "runs", label)
    os.makedirs(out, exist_ok=True)
    shutil.copy2(res, out)
    log = os.path.join(os.path.dirname(res), "run.log")
    if os.path.exists(log):
        shutil.copy2(log, out)
    for pat in FIGURE_FILES.get(label, []):
        for f in glob.glob(os.path.join(os.path.dirname(res), pat)):
            shutil.copy2(f, out)
    n += 1
for f in ("results.csv", "robustness.csv", "results.xlsx"):
    p = os.path.join(a.root, f)
    if os.path.exists(p):
        shutil.copy2(p, a.dest)
os.makedirs(os.path.join(a.dest, "maddpg"), exist_ok=True)
for d in a.maddpg_meta:
    m = os.path.join(d, "metadata.json")
    if os.path.exists(m):
        shutil.copy2(m, os.path.join(a.dest, "maddpg", f"{os.path.basename(d.rstrip('/'))}_metadata.json"))
print(f"copied {n} runs to {os.path.join(a.dest, 'runs')}")
