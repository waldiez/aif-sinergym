# One Agent per Zone: Decentralised Active Inference for Multi-Zone HVAC Control

Code, configurations and results for the paper *One Agent per Zone:
Decentralised Active Inference under Partial Observability for Multi-Zone HVAC
Control on Edge Hardware* (P. Kasnesis, L. Toumanidis, C. Chatzigeorgiou,
A. Contiero Syropoulou, M. De Prado).

Each of the 15 conditioned zones of the ASHRAE 90.1 OfficeMedium prototype is
controlled by an independent active-inference (AIF) agent. The agents exchange
no messages; each one reads the metered power of the other two air loops as a
congestion signal. The repository contains the agents, the rule-based and
MADDPG baselines, the experiment runner, and the scripts that produce every
table and figure of the paper.

## Main result

Five noisy weather years, unseen during training (mean over years):

| Controller | Peak (kW) | CF | CS<sub>occ</sub> | Energy (MWh) | Cost (k$) |
|---|---|---|---|---|---|
| AIF (fixed reference) | **88.3** | **0.929** | 0.912 | 117.4 | 55.4 |
| RBC reactive | 93.8 | 0.974 | 0.916 | **106.2** | **53.7** |
| RBC proactive | 90.9 | 0.958 | **0.946** | 139.2 | 61.1 |
| MADDPG + floor meters | 97.0 | 0.936 | 0.926 | 119.6 | 55.9 |

The full tables, including the deterministic year, the unseen climates and the
ablations, are in the paper and in `results/`.

## Repository layout

```
code/
  aif_agent.py            AIF agents (batched PyTorch) + simulation runner
  register_env.py         Sinergym environment (per-zone setpoints, floor meters, weather noise)
  metrics_utils.py        shared scoring: comfort, deadband, CF, tariff cost, MADDPG reward
  proactive_rbc.py        proactive rule-based controller
  rbc_office_reactive.py  reactive rule-based controller (same code base, two switches)
  rbc_v2.py               RBC runner, demand-limiting RBC, shared evaluation loop
  maddpg_v4.py            MADDPG/TD3 as trained for the cold and warm-start fleets
  maddpg_v5.py            same, plus --floor-features and self-contained checkpoints
  maddpg_eval.py          deterministic MADDPG evaluation under the same protocol
  run_ablation_v4.py      experiment runner: all suites, reports, robustness table
  paired_stats.py         per-year paired comparisons (Table II)
  make_figures.py         figures and LaTeX tables from the result files
  preflight.py            checks on the real environment before running
  tests/                  regression tests (no EnergyPlus needed)
building/                 custom epJSON model (see building/README.md)
scripts/                  install_building.py, reproduce_all.sh, collect_results.py
results/                  per-run results.json, summary CSVs, MADDPG training curves
figures/                  generated figures
paper/                    LaTeX source, bibliography and figures
env/                      software versions used for the paper
```

## Setup

The results in the paper were produced with Python 3.12.7, Sinergym 3.11.0,
EnergyPlus 24.1.0, PyTorch 2.7.1 (CUDA 12.6) on an NVIDIA RTX 3090
(`env/env_info.txt`; the exact package list is in `env/requirements-lock.txt`).

1. Install EnergyPlus 24.1.0 and Sinergym 3.11.0 following the Sinergym
   documentation, then the remaining packages:

       pip install -r requirements.txt

2. Install the custom building model into Sinergym:

       python scripts/install_building.py

3. Check the installation (about 15 minutes):

       cd code
       python tests/test_agent.py
       python tests/test_runner.py
       python preflight.py --legacy

## Reproducing the results

`scripts/reproduce_all.sh` runs every experiment in order, with the exact
settings used for the paper, then builds the tables and figures and copies the
result files into `results/`. All steps are resumable, so the script can be
interrupted and restarted. On one RTX 3090 the AIF and RBC experiments take
about 10 hours with three parallel jobs, and each MADDPG fleet takes about two
days to train.

To regenerate the tables and figures from the stored results without running
any simulation, see `results/README.md`.

### Where each result comes from

| Paper | Runner suites | Output |
|---|---|---|
| Table I, Table II, Fig. 1 (noisy years W1) | `transfer`, `rbc_transfer`, `maddpg_transfer` | `results/robustness.csv`, `paired_stats.py` |
| Table III (deterministic year W0) | `core`, `alt`, `rbc`, `maddpg` | `results/results.csv` |
| Table IV (unseen climates W2) | `transfer`, `rbc_transfer`, `maddpg_transfer` | `results/robustness.csv` |
| Table V (ablations, mechanism) | `core`, `sweep`, `horizon`, `alt`, `mechanism`, `followup` | `results/results.csv` |
| Fig. 2 (design day) | `alt_fixedref`, `rbc_proactive` | `make_figures.py` |
| Fig. 3 (tariff) | W1 runs | `make_figures.py` |

## Notes on reproducibility

* All runs are deterministic: the AIF agents select actions by argmax and the
  weather is either the fixed TMY3 year or seeded noise. Rerunning a
  configuration gives the same result on the same hardware.
* AIF runs on the GPU in float32 by default. Running on a CPU, or in float64,
  can change near-tied decisions; keep all runs of a comparison on the same
  device.
* The noisy weather of a given seed is identical for every controller; the
  report checks this by comparing the logged outdoor temperatures.
* `code/tests/aif_agent_v14_reference.py` is the earlier version of the agent.
  `aif_agent.py --legacy_infer --horizon_ref 0` reproduces it exactly, which the
  tests and `preflight.py --legacy` verify.

## License

Apache License 2.0.

## Acknowledgment

This work was carried out in the SYNAPSE project, funded through the third open
call of dAIEDGE, the European Network of Excellence for distributed,
trustworthy, efficient and scalable AI at the Edge.
