#!/usr/bin/env python3
"""
make_figures.py -- paper figures and LaTeX tables straight from ab_runs/.

  python make_figures.py --root ab_runs --out figs \
      --cold_meta checkpoint-cold/metadata.json \
      --floor_meta checkpoint-cold-floor/metadata.json

Figures (PDF + PNG):
  fig_pareto_w1    occupied-only CS vs total cost / sum of monthly peaks,
                   mean +- std over the 5 noisy test years (W1)
  fig_tariff       mean total cost vs demand-charge rate (W1); break-even points
  fig_peakday      per-floor HVAC power on the design day, AIF vs RBC proactive (W0)
  fig_energy_w     per-agent learned energy weight over the 5 adaptation years
  fig_training     MADDPG training curves, with / without floor power
Tables (.tex, booktabs):
  tab_w1           robustness table (W1, mean +- std)
  tab_w2           unseen climates (hot / cool)
  tab_w0           deterministic year (AIF: first year and last-3 mean)
  tab_ablation     AIF ablation + mechanism controls (W0, last-3 mean)
Every number comes from results.json; missing runs are skipped with a note.
"""
import argparse, csv, glob, json, os
import numpy as np

SEEDS = [1, 2, 3, 4, 5]
TARIFF = 26.0   # $/kW-month in the evaluation tariff

# (label shown, run-label prefix before _var_s{n} / _hot / _cool, family)
CONTROLLERS = [
    ("AIF (fixed ref.)", "tr_frz_alt_fixedref", "aif"),
    ("AIF (running ref.)", "tr_frz_04_full", "aif"),
    ("AIF, no congestion", "tr_frz_01_baseline", "aif"),
    ("RBC reactive", "tr_rbc_reactive", "rbc"),
    ("RBC proactive", "tr_rbc_proactive", "rbc"),
    ("RBC demand-limiting", "tr_rbc_dl", "rbc"),
    ("MADDPG cold", "tr_mad_cold", "mad"),
    ("MADDPG warm (reactive)", "tr_mad_warm_rea", "mad"),
    ("MADDPG + floor power", "tr_mad_cold_floor", "mad"),
]
AIF_SWEEP = [("no cong.", "01_baseline"), ("learn_C", "02_learnC"), ("c5", "03_cong5"),
             ("c3+L", "sw_c3_L1"), ("c4+L", "sw_c4_L1"), ("c5+L", "04_full"),
             ("c5+L fixed", "alt_fixedref"), ("c8", "sw_c8_L0"), ("c8+L", "sw_c8_L1")]
RBC_CURVE = [("0.5", "pro_cm0p5"), ("1.0", "proactive"), ("1.5", "pro_cm1p5"),
             ("2.0", "pro_cm2p0"), ("2.5", "pro_cm2p5")]
STYLE = {"aif": dict(color="#1f77b4", marker="o"), "rbc": dict(color="#d62728", marker="s"),
         "mad": dict(color="#2ca02c", marker="^")}


def eps(root, label):
    p = os.path.join(root, label, "results.json")
    if not os.path.exists(p):
        return None
    e = json.load(open(p)).get("episodes", [])
    return e or None


def w1(root, prefix, key):
    v = []
    for s in SEEDS:
        e = eps(root, f"{prefix}_var_s{s}")
        if e and e[0].get(key) is not None:
            v.append(e[0][key])
    return np.array(v)


def ms(a):
    return (float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0) if len(a) else (None, None)


# ---------------------------------------------------------------- figures
AIF_CURVE = [("0", "02_learnC"), ("3", "sw_c3_L1"), ("4", "sw_c4_L1"),
             ("5", "alt_fixedref"), ("8", "sw_c8_L1")]       # congestion weight, adaptive weights on


def _curve(ax, root, prefix_fmt, items, key, sc, fam, fill_label=None):
    pts = []
    for tag, src in items:
        x, xs = ms(w1(root, prefix_fmt.format(src), "CS_occ"))
        y, ys = ms(w1(root, prefix_fmt.format(src), key))
        if x is None:
            continue
        filled = (src == fill_label)
        ax.errorbar(x, y * sc, xerr=xs, yerr=ys * sc, fmt=STYLE[fam]["marker"], ms=4.5,
                    color=STYLE[fam]["color"], mfc=STYLE[fam]["color"] if filled else "white",
                    elinewidth=0.6, capsize=1.5, zorder=3)
        if fam == "aif" or tag in ("0.5", "2.5"):          # RBC: label the ends only
            ax.annotate(tag if fam == "aif" else f"{tag}$^\\circ$C", (x, y * sc), fontsize=6,
                        color=STYLE[fam]["color"], xytext=(4, -9 if fam == "aif" else 4),
                        textcoords="offset points")
        pts.append((x, y * sc))
    if pts:   # joined in knob order, not sorted by comfort
        ax.plot(*zip(*pts), "-", color=STYLE[fam]["color"], lw=0.9, alpha=0.6, zorder=2)


def fig_pareto(root, out, plt):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, key, ylab in ((axs[0], "tot_cost", "total cost (k$ / yr)"),
                          (axs[1], "sum_monthly_peaks_kW", "sum of monthly peaks (kW)")):
        sc = 1e-3 if key == "tot_cost" else 1.0
        _curve(ax, root, "tr_frz_{}", AIF_CURVE, key, sc, "aif", fill_label="alt_fixedref")
        _curve(ax, root, "tr_rbc_{}", RBC_CURVE, key, sc, "rbc")
        for lab, pre, fam in CONTROLLERS:
            if fam == "aif" or pre in ("tr_rbc_proactive",):
                continue
            x, xs = ms(w1(root, pre, "CS_occ")); y, ys = ms(w1(root, pre, key))
            if x is None:
                continue
            ax.errorbar(x, y * sc, xerr=xs, yerr=ys * sc, fmt=STYLE[fam]["marker"], ms=5,
                        color=STYLE[fam]["color"], elinewidth=0.6, capsize=1.5, zorder=3)
            off = (-52, 4) if "demand" in lab else (3, 3)
            ax.annotate(lab.replace("MADDPG", "M.").replace("RBC ", ""), (x, y * sc),
                        fontsize=6, xytext=off, textcoords="offset points")
        ax.set_xlabel("occupied-hours comfort score (CS$_{occ}$)"); ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, lw=0.4)
    axs[0].plot([], [], "o-", color=STYLE["aif"]["color"], mfc="white",
                label="AIF, congestion weight 0 to 8")
    axs[0].plot([], [], "s-", color=STYLE["rbc"]["color"], mfc="white",
                label="RBC proactive, cooling margin 0.5 to 2.5 $^\\circ$C")
    axs[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout(); save(fig, out, "fig_pareto_w1")


def fig_tariff(root, out, plt):
    rates = np.linspace(0, 100, 201)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    curves = {}
    for lab, pre, fam in CONTROLLERS:
        en = w1(root, pre, "en_cost"); pk = w1(root, pre, "sum_monthly_peaks_kW")
        if len(en) == 0 or len(en) != len(pk):
            continue
        c = (en[:, None] + rates[None, :] * pk[:, None]).mean(0) / 1e3
        curves[lab] = (en.mean(), pk.mean())
        if lab in ("AIF (fixed ref.)", "RBC reactive", "RBC proactive", "MADDPG + floor power"):
            ax.plot(rates, c, color=STYLE[fam]["color"],
                    ls={"aif": "-", "rbc": "--", "mad": ":"}[fam] if lab != "RBC proactive" else "-.",
                    lw=1.2, label=lab)
    ax.axvline(TARIFF, color="k", lw=0.6, ls=":"); ax.text(TARIFF + 1, ax.get_ylim()[0], "evaluated\ntariff",
                                                           fontsize=6, va="bottom")
    ax.set_xlabel("demand charge ($ / kW-month)"); ax.set_ylabel("mean total cost (k$ / yr)")
    ax.legend(fontsize=6); ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(); save(fig, out, "fig_tariff")
    a, b = curves.get("AIF (fixed ref.)"), curves.get("RBC reactive")
    if a and b and a[1] < b[1]:
        print(f"  break-even demand charge AIF vs RBC reactive: "
              f"{(a[0] - b[0]) / (b[1] - a[1]):.1f} $/kW-month")


def floor_series(root, label, ep_file):
    """[N,3] floor kW and (month, day, hour) per step, for AIF and RBC logs."""
    p = os.path.join(root, label, ep_file)
    if not os.path.exists(p):
        return None
    a = np.load(p)
    if a.shape[1] == 6:                       # AIF: month, day, hour, kW x3
        return a[:, :3].astype(int), a[:, 3:6]
    z = os.path.join(root, label, "zone_temps_actions.csv")   # RBC/MADDPG: W x3
    with open(z) as fh:
        rd = csv.reader(fh); next(rd)
        t = np.array([[int(r[2]), int(r[3]), int(r[4])] for r in rd if r[0] == "1"])
    t = np.vstack([t[1:], t[-1:]])            # post-step time of each row
    return t[:len(a)], a / 1000.0


def fig_peakday(root, out, plt, day):
    m, d = [int(x) for x in day.split("/")]
    runs = [("AIF (fixed ref.)", "alt_fixedref", None), ("RBC proactive", "rbc_proactive", "floorlog_ep1.npy")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, (lab, run, f) in zip(axs, runs):
        if f is None:
            cands = sorted(glob.glob(os.path.join(root, run, "floorlog_ep*.npy")))
            f = os.path.basename(cands[-1]) if cands else "x"
        s = floor_series(root, run, f)
        if s is None:
            ax.set_title(f"{lab}: missing"); continue
        t, kw = s
        sel = (t[:, 0] == m) & (t[:, 1] == d)
        if not sel.any():
            ax.set_title(f"{lab}: no data for {day}", fontsize=8); continue
        x = np.arange(sel.sum()) / 4.0
        for j, nm in enumerate(("bottom", "middle", "top")):
            ax.plot(x, kw[sel, j], lw=1.0, label=f"{nm} floor")
        ax.plot(x, kw[sel].sum(1), "k", lw=1.4, label="building")
        ax.set_title(f"{lab}  (peak {kw[sel].sum(1).max():.1f} kW)", fontsize=8)
        ax.set_xlabel(f"hour of {day}"); ax.set_xlim(0, 24); ax.grid(alpha=0.3, lw=0.4)
    axs[0].set_ylabel("HVAC power (kW)"); axs[0].legend(fontsize=6)
    fig.tight_layout(); save(fig, out, "fig_peakday")


def fig_energy_w(root, out, plt, run="alt_fixedref"):
    files = sorted(glob.glob(os.path.join(root, run, "energy_w_daily_ep*.npy")),
                   key=lambda p: int(p.rsplit("ep", 1)[1].split(".")[0]))
    if not files:
        print(f"  fig_energy_w: no energy_w logs in {run}"); return
    w = np.vstack([np.load(f) for f in files])          # days x agents
    order = [0, 3, 4, 5, 6, 1, 7, 8, 9, 10, 2, 11, 12, 13, 14]   # grouped by floor
    names = ["Cb", "Pb1", "Pb2", "Pb3", "Pb4", "Cm", "Pm1", "Pm2", "Pm3", "Pm4",
             "Ct", "Pt1", "Pt2", "Pt3", "Pt4"]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    im = ax.imshow(w[:, order].T, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   interpolation="nearest", extent=[0, len(w) / 365.0, 14.5, -0.5])
    ax.set_yticks(range(15)); ax.set_yticklabels(names, fontsize=5)
    for y in (4.5, 9.5):
        ax.axhline(y, color="w", lw=0.6)
    ax.set_xlabel("adaptation year")
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("energy weight $w^E_i$")
    at_bound = float(np.mean((w < 0.05) | (w > 0.95)))
    print(f"  energy_w: {100*at_bound:.0f}% of daily snapshots at a bound")
    fig.tight_layout(); save(fig, out, "fig_energy_w")


def fig_training(out, plt, metas):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for (lab, path), ls in zip(metas, ("-", "--")):
        if not path or not os.path.exists(path):
            continue
        h = json.load(open(path))["metrics_history"]
        x = np.arange(1, len(h["rewards"]) + 1)
        axs[0].plot(x, h["rewards"], ls, label=lab)
        axs[1].plot(x, h["cs_occ"], ls); axs[2].plot(x, np.array(h["energy"]) / 1e3, ls)
    for ax, t in zip(axs, ("shaped reward", "CS$_{occ}$", "energy (MWh)")):
        ax.set_xlabel("training year"); ax.set_title(t, fontsize=8); ax.grid(alpha=0.3, lw=0.4)
    axs[1].set_ylim(0.85, 0.94); axs[0].legend(fontsize=6)
    fig.tight_layout(); save(fig, out, "fig_training")


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    print(f"  wrote {name}")


# ---------------------------------------------------------------- tables
def pm(a, fmt):
    m, s = ms(a)
    return "--" if m is None else f"{format(m, fmt)}$\\pm${format(s, fmt)}"


def tab_w1(root, out):
    rows = []
    for lab, pre, fam in CONTROLLERS:
        if len(w1(root, pre, "peak_kW")) == 0:
            continue
        rows.append(" & ".join([lab, pm(w1(root, pre, "peak_kW"), ".1f"),
                                pm(w1(root, pre, "sum_monthly_peaks_kW"), ".0f"),
                                pm(w1(root, pre, "CF"), ".3f"), pm(w1(root, pre, "CS_occ"), ".3f"),
                                pm(w1(root, pre, "ZCR"), ".1f"),
                                pm(w1(root, pre, "energy_kWh") / 1e3, ".1f"),
                                pm(w1(root, pre, "tot_cost") / 1e3, ".1f")]) + r" \\")
    write_tab(out, "tab_w1", "Robustness on five noisy test years (W1, mean$\\pm$std).",
              "l" + "c" * 7, ["Controller", "Peak (kW)", "$\\Sigma$ monthly peaks (kW)", "CF",
                              "CS$_{occ}$", "ZCR (\\%)", "Energy (MWh)", "Cost (k\\$)"], rows)


def tab_w2(root, out):
    rows = []
    for lab, pre, fam in CONTROLLERS:
        cells = [lab]
        for w in ("hot", "cool"):
            e = eps(root, f"{pre}_{w}")
            cells += (["--"] * 3 if not e else
                      [f"{e[0]['peak_kW']:.1f}", f"{e[0]['CS_occ']:.3f}", f"{e[0]['tot_cost']/1e3:.1f}"])
        if any(c != "--" for c in cells[1:]):
            rows.append(" & ".join(cells) + r" \\")
    write_tab(out, "tab_w2", "Unseen climates (W2, one deterministic year each).", "l" + "c" * 6,
              ["Controller", "Hot: peak", "CS$_{occ}$", "k\\$", "Cool: peak", "CS$_{occ}$", "k\\$"], rows)


W0_ROWS = [("RBC reactive", "rbc_reactive"), ("RBC proactive", "rbc_proactive"),
           ("RBC demand-limiting", "rbc_dl"), ("MADDPG cold", "maddpg_cold"),
           ("MADDPG warm (reactive)", "maddpg_warm_rea"), ("MADDPG + floor power", "maddpg_cold_floor"),
           ("AIF, no congestion", "01_baseline"), ("AIF (running ref.)", "04_full"),
           ("AIF (fixed ref.)", "alt_fixedref")]


def last3(e, k):
    v = [x.get(k) for x in e[-3:] if x.get(k) is not None]
    return float(np.mean(v)) if v else None


def tab_w0(root, out):
    rows = []
    for lab, run in W0_ROWS:
        e = eps(root, run)
        if not e:
            continue
        f = lambda k, fmt, sc=1.0: "--" if last3(e, k) is None else format(last3(e, k) * sc, fmt)
        first = (f"{e[0]['peak_kW']:.1f} / {e[0].get('CF', float('nan')):.3f}"
                 if len(e) > 1 else "")
        rows.append(" & ".join([lab, f("peak_kW", ".1f"), f("sum_monthly_peaks_kW", ".0f"),
                                f("CF", ".3f"), f("CS_occ", ".3f"), f("ZCR", ".1f"),
                                f("db_viol", ".1f"), f("energy_kWh", ".1f", 1e-3),
                                f("tot_cost", ".1f", 1e-3), first]) + r" \\")
    write_tab(out, "tab_w0", "Deterministic mixed year (W0). AIF: mean of the last three of five "
              "adaptation years; last column: its first year.", "l" + "c" * 9,
              ["Controller", "Peak", "$\\Sigma$ m.p.", "CF", "CS$_{occ}$", "ZCR", "DB viol.",
               "MWh", "k\\$", "Year 1 peak / CF"], rows)


ABL_ROWS = [("no congestion, no learn\\_C", "01_baseline"), ("learn\\_C only", "02_learnC"),
            ("congestion 5 only", "03_cong5"), ("congestion 2 + learn\\_C", "05_cong2_learnC"),
            ("congestion 4 + learn\\_C", "sw_c4_L1"), ("\\textbf{congestion 5 + learn\\_C}", "04_full"),
            ("\\quad fixed normaliser", "alt_fixedref"), ("\\quad v14 inference (legacy)", "00_full_legacy_v14"),
            ("\\quad horizon 4", "h04_full"), ("\\quad horizon 16", "h16_full"),
            ("\\quad stochastic actions", "alt_stoch05"), ("\\quad TOU on", "alt_tou1"),
            ("\\quad coupling (learned)", "alt_couple"),
            ("signal: own floor (c5 / c8)", "mech_own|mech_own_c8"),
            ("signal: whole building (c5 / c8)", "mech_building|mech_building_c8"),
            ("signal: 24\\,h lagged (c5 / c8)", "mech_lag24|mech_lag24_c8"),
            ("signal: constant (c5 / c8)", "mech_const|mech_const_c8")]


def tab_ablation(root, out):
    rows = []
    for lab, run in ABL_ROWS:
        parts = run.split("|")
        cells = [lab]
        for k, fmt in (("peak_kW", ".1f"), ("CF", ".3f"), ("CS_occ", ".3f"), ("tot_cost", ".1f")):
            vals = []
            for r in parts:
                e = eps(root, r)
                v = last3(e, k) if e else None
                vals.append("--" if v is None else format(v / (1e3 if k == "tot_cost" else 1), fmt))
            cells.append(" / ".join(vals))
        rows.append(" & ".join(cells) + r" \\")
    hetero = [last3(eps(root, f"mech_hetero_s{s}"), "peak_kW") for s in range(8)
              if eps(root, f"mech_hetero_s{s}")]
    note = ""
    if hetero:
        ok = sum(1 for h in hetero if h < 88.0)
        note = (f" Fixed random per-agent energy weights with congestion 5 reach the low-peak "
                f"regime in {ok} of {len(hetero)} draws.")
    write_tab(out, "tab_ablation", "AIF ablation and mechanism controls (W0, mean of last three "
              "years)." + note, "lcccc", ["Configuration", "Peak (kW)", "CF", "CS$_{occ}$", "Cost (k\\$)"],
              rows)


def write_tab(out, name, caption, cols, header, rows):
    with open(os.path.join(out, f"{name}.tex"), "w") as f:
        f.write("\\begin{table*}[t]\n\\centering\\footnotesize\n")
        f.write(f"\\caption{{{caption}}}\\label{{{name}}}\n")
        f.write(f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n" + " & ".join(header) + r" \\" + "\n\\midrule\n")
        f.write("\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print(f"  wrote {name}.tex ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ab_runs"); ap.add_argument("--out", default="figs")
    ap.add_argument("--cold_meta", default="checkpoint-cold/metadata.json")
    ap.add_argument("--floor_meta", default="checkpoint-cold-floor/metadata.json")
    ap.add_argument("--peak_day", default="8/5")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "font.family": "serif"})
    os.makedirs(a.out, exist_ok=True)
    for fn in (lambda: fig_pareto(a.root, a.out, plt), lambda: fig_tariff(a.root, a.out, plt),
               lambda: fig_peakday(a.root, a.out, plt, a.peak_day),
               lambda: fig_energy_w(a.root, a.out, plt),
               lambda: fig_training(a.out, plt, [("cold", a.cold_meta),
                                                 ("cold + floor power", a.floor_meta)]),
               lambda: tab_w1(a.root, a.out), lambda: tab_w2(a.root, a.out),
               lambda: tab_w0(a.root, a.out), lambda: tab_ablation(a.root, a.out)):
        try:
            fn()
        except Exception as ex:      # one failing figure must not stop the rest
            print(f"  skipped: {ex!r}")


if __name__ == "__main__":
    main()
