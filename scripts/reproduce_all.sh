#!/usr/bin/env bash
# Reproduce every result in the paper. Run from the repository root.
# Approximate times on one RTX 3090 with --jobs 3 (AIF: ~2 min per simulated
# year; RBC and MADDPG evaluation: ~1 to 6 min per year; MADDPG training: ~2
# days per fleet). Every step is resumable: finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/../code"
J="--jobs 3 --chdir --clean_eplus"

# 0. Checks (about 15 min): observation layout, calendar, weather noise, v14 reproduction
python3 tests/test_agent.py
python3 tests/test_runner.py
python3 preflight.py --legacy

# 1. AIF on the deterministic year W0 (about 6 h)
python3 run_ablation_v4.py --suite core                                   # Table V core rows
python3 run_ablation_v4.py --suite sweep,horizon,alt,mechanism,followup $J # Table V, Fig. 1

# 2. AIF on the noisy years W1 and the unseen climates W2 (about 3 h)
python3 run_ablation_v4.py --suite transfer $J \
  --freeze_from 01_baseline,04_full,alt_fixedref,sw_c8_L1,mech_const,02_learnC,03_cong5,sw_c3_L1,sw_c4_L1,sw_c8_L0 \
  --online_from 01_baseline,04_full,alt_fixedref

# 3. Rule-based controllers on W0, W1 and W2 (about 30 min)
python3 run_ablation_v4.py --suite rbc,rbc_pareto $J
python3 run_ablation_v4.py --suite rbc_transfer $J \
  --rbc_from rbc_reactive,rbc_proactive,rbc_dl,rbc_pro_cm0p5,rbc_pro_cm1p5,rbc_pro_cm2p0,rbc_pro_cm2p5

# 4. MADDPG training (about 2 days per fleet; maddpg_v5 without --floor-features
#    is identical to maddpg_v4 except that every file is written inside
#    --checkpoint-dir)
python3 maddpg_v5.py --mode train --episodes 50 --no-warmup --checkpoint-dir checkpoint-cold
python3 maddpg_v5.py --mode train --episodes 50 --warmup_rbc reactive --checkpoint-dir checkpoint-warm-reactive
python3 maddpg_v5.py --mode train --episodes 50 --no-warmup --floor-features --checkpoint-dir checkpoint-cold-floor

# 5. MADDPG evaluation on W0, W1 and W2 (about 1 h)
M="cold=checkpoint-cold,warm_rea=checkpoint-warm-reactive,cold_floor=checkpoint-cold-floor"
python3 run_ablation_v4.py --suite maddpg,maddpg_transfer $J --maddpg "$M"

# 6. Tables, paired statistics and figures
python3 run_ablation_v4.py --suite all --report --maddpg "$M" \
  --freeze_from 01_baseline,04_full,alt_fixedref,sw_c8_L1,mech_const,02_learnC,03_cong5,sw_c3_L1,sw_c4_L1,sw_c8_L0 \
  --online_from 01_baseline,04_full,alt_fixedref \
  --rbc_from rbc_reactive,rbc_proactive,rbc_dl,rbc_pro_cm0p5,rbc_pro_cm1p5,rbc_pro_cm2p0,rbc_pro_cm2p5
python3 paired_stats.py --a frz:alt_fixedref --b rbc:reactive rbc:proactive mad:cold_floor
python3 make_figures.py --root ab_runs --out ../figures \
  --cold_meta checkpoint-cold/metadata.json --floor_meta checkpoint-cold-floor/metadata.json

# 7. Copy the small result files into results/ (for the repository)
python3 ../scripts/collect_results.py --root ab_runs --dest ../results \
  --maddpg_meta checkpoint-cold checkpoint-warm-reactive checkpoint-cold-floor
