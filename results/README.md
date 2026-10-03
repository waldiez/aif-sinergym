# Results

Filled by `scripts/collect_results.py` (last step of `scripts/reproduce_all.sh`).

* `runs/<label>/results.json`: per-episode metrics of every run (peak, monthly
  peaks, CF, comfort scores, deadband violations, energy, cost, the annual peak
  event, and for AIF the override rate, energy weights and congestion signal).
* `runs/<label>/run.log`: the exact command and console output of the run.
* `results.csv` and `robustness.csv`: the summary tables printed by the runner.
* `maddpg/*_metadata.json`: MADDPG training curves (50 simulated years).

Run labels: `tr_frz_*` frozen AIF, `tr_onl_*` online AIF, `tr_rbc_*` rule-based,
`tr_mad_*` MADDPG; suffix `_var_s1..5` = noisy year W1 with that seed,
`_hot` / `_cool` = unseen climates W2. Labels without `tr_` are runs on the
deterministic year W0 (AIF: five adaptation years).

To regenerate every table and figure without simulating:

    cd code
    python3 make_figures.py --root ../results/runs --out ../figures \
      --cold_meta ../results/maddpg/checkpoint-cold_metadata.json \
      --floor_meta ../results/maddpg/checkpoint-cold-floor_metadata.json
