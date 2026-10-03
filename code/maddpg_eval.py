#!/usr/bin/env python3
"""
maddpg_eval.py -- evaluate a trained MADDPG checkpoint under the SAME weather
protocol, scoring and results.json schema as the AIF and RBC runs.

Deterministic actors (no exploration noise), weights and normaliser loaded
from maddpg_v4 files; the replay buffer is NOT loaded. The action pipeline is
identical to maddpg_v4.evaluate(): ObservationBuilder -> AgentSequenceBuilder
-> agent.select_action(seqs, normalizer, add_noise=False).

  python maddpg_eval.py --checkpoint_dir checkpoint-cold --out_dir ab_runs/maddpg_cold
  python maddpg_eval.py --model maddpg_office_ep50.pt --normalizer maddpg_normalizer.npz ...
  python maddpg_eval.py --checkpoint_dir checkpoint-cold --weather_variability 1.5 --seed 3 ...
"""
import argparse, importlib, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rbc_v2 import run_episode  # noqa: E402  (shared scoring loop)


class MADDPGPolicy:
    def __init__(self, m, agent, normalizer):
        self.m, self.agent, self.norm = m, agent, normalizer
        self.ob = m.ObservationBuilder(); self.ob.reset()
        self.seq = m.AgentSequenceBuilder(m.N_OCCUPIED, m.Config.SEQ_LEN, m.Config.AGENT_OBS_DIM)
        self.first = True

    def get_action(self, obs, info=None):
        ao = self.ob.build_agent_obs(obs)
        if self.first:
            self.seq.reset(ao); cur = self.seq.get_current(); self.first = False
        else:
            cur = self.seq.append(ao)
        return self.agent.select_action(cur, self.norm, add_noise=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", default=None,
                    help="maddpg_v4 checkpoint dir (model.pt + normalizer.npz)")
    ap.add_argument("--model", default=None, help="explicit actor/critic .pt (overrides dir)")
    ap.add_argument("--normalizer", default=None, help="explicit normalizer .npz")
    ap.add_argument("--maddpg_module", default="maddpg_v4")
    ap.add_argument("--weather", default="mixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weather_variability", type=float, default=0.0)
    ap.add_argument("--floor_power_idx", default="92,93,94")
    ap.add_argument("--floor_power_scale", type=float, default=0.0011111)
    ap.add_argument("--device", default=None, help="cpu | cuda (default: maddpg_v4 Config)")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--results_json", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--episodes", type=int, default=1, help="ignored: deterministic policy")
    args = ap.parse_args()

    model = args.model or (os.path.join(args.checkpoint_dir, "model.pt") if args.checkpoint_dir else None)
    norm = args.normalizer or (os.path.join(args.checkpoint_dir, "normalizer.npz")
                               if args.checkpoint_dir else None)
    for p in (model, norm):
        if not p or not os.path.exists(p):
            raise SystemExit(f"missing file: {p}")

    # 32-dim normaliser -> checkpoint trained with floor-power features (maddpg_v5)
    obs_dim = int(np.load(norm)["mean"].shape[0])
    if obs_dim != 29 and args.maddpg_module == "maddpg_v4":
        args.maddpg_module = "maddpg_v5"
    m = importlib.import_module(args.maddpg_module)
    if hasattr(m, "enable_floor_features"):
        m.enable_floor_features(obs_dim == 32)
    assert m.Config.AGENT_OBS_DIM == obs_dim, \
        f"normaliser has {obs_dim} features, model expects {m.Config.AGENT_OBS_DIM}"
    if args.device:
        import torch
        m.Config.DEVICE = torch.device(args.device)
    from register_env import make_custom_env
    from metrics_utils import CustomRewardWrapper

    np.random.seed(int(args.seed))       # same seeding order as AIF / RBC runs
    kw = ({"weather_variability": float(args.weather_variability)}
          if args.weather_variability and args.weather_variability > 0 else {})
    env = CustomRewardWrapper(make_custom_env(weather=args.weather, real_world=False, **kw))

    m.Config.BUFFER_SIZE = 1024          # evaluation never touches the replay buffer
    agent = m.MADDPGAgent(env)
    agent.load(model)
    normalizer = m.AgentObsNormalizer(m.Config.AGENT_OBS_DIM)
    normalizer.load(norm)
    if not normalizer.frozen:
        normalizer.freeze()
    args.model, args.normalizer = model, norm
    run_episode(env, MADDPGPolicy(m, agent, normalizer), args, "maddpg_eval.py",
                f"MADDPG[{os.path.basename(os.path.dirname(os.path.abspath(model)))}/"
                f"{os.path.basename(model)}]")


if __name__ == "__main__":
    main()
