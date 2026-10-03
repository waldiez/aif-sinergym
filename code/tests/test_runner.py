"""
End-to-end test of aif_agent.run_simulation with a fake
Sinergym environment (toy plant, 95-dim obs, ~2 weeks per 'year').
Checks the results JSON, the out_dir artefacts and explicit checkpoint loading.

  python tests/test_runner.py
"""
import json, os, sys, tempfile, types
import numpy as np
import gymnasium as gym

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_agent import ToyPlant, v15, N  # noqa: E402


class FakeEplus(gym.Env):
    STEPS = 14 * 96                    # two weeks, crosses a month boundary

    def __init__(self, weather_variability=0.0):
        self.action_space = gym.spaces.Box(
            low=np.array([15.0] * N + [22.5] * N, dtype=np.float32),
            high=np.array([22.5] * N + [30.0] * N, dtype=np.float32))
        self.observation_space = gym.spaces.Box(-1e9, 1e9, (95,), np.float32)
        self.sigma = weather_variability

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = ToyPlant(seed=0 if seed is None else seed)
        self.fp = np.zeros(3); self.t = 0
        return self._obs(), {}

    def _obs(self):
        o = self.plant.obs(self.fp)
        o[90] = self.fp.sum() * 1.05            # facility HVAC demand (W)
        o[45:60] = 1.0 if 7 <= self.plant.h <= 19 else 0.0
        return o

    def step(self, action):
        self.fp = self.plant.step(np.asarray(action, dtype=np.float64))
        self.t += 1
        return self._obs(), 0.0, False, self.t >= self.STEPS, {}


fake = types.ModuleType("register_env")
fake.make_custom_env = lambda weather="mixed", real_world=False, **kw: FakeEplus(**kw)
sys.modules["register_env"] = fake

COMMON = dict(mode="zone", structural=True, energy_weight=0.2, tou_weight=0.0,
              congestion_weight=5.0, action_temp=0.0, policy_len=8, learn_C=True,
              c_lr=0.02, energy_w_min=0.0, energy_w_max=1.0, deadband_weight=8.0,
              floor_power_idx=[92, 93, 94], floor_power_scale=0.0011111,
              device="cpu", dtype="double", horizon_ref=8)


def main():
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a")
        v15.run_simulation(episodes=2, out_dir=a, checkpoint=os.path.join(a, "ck.pkl"),
                           label="t_full", **COMMON)
        r = json.load(open(os.path.join(a, "results.json")))
        assert r["started_fresh"] and len(r["episodes"]) == 2
        e = r["episodes"][-1]
        for k in ["peak_kW", "CF", "CS", "CS_occ", "ZCR", "tot_cost", "monthly_peak_kW",
                  "peak_event", "override_rate", "energy_w_final", "cong_obs_mean_by_floor"]:
            assert k in e and e[k] is not None, k
        assert len(e["monthly_peak_kW"]) == 2, e["monthly_peak_kW"]
        for f in ["zone_temps_actions.csv", "floorlog_ep1.npy", "energy_w_daily_ep1.npy", "ck.pkl"]:
            assert os.path.exists(os.path.join(a, f)), f
        assert not os.path.exists(os.path.join(a, "obs.csv")), "debug logs must be opt-in"
        print(f"  ok  results.json: peak={e['peak_kW']:.1f} CF={e['CF']:.3f} "
              f"CS={e['CS']:.3f} override={100*e['override_rate']:.2f}% "
              f"peak_event={e['peak_event']['month']}/{e['peak_event']['day']} "
              f"h{e['peak_event']['hour']}")

        # a second run at the SAME save path must not silently load it
        v15.run_simulation(episodes=1, out_dir=a, checkpoint=os.path.join(a, "ck.pkl"),
                           results_json=os.path.join(a, "r2.json"), **COMMON)
        r2 = json.load(open(os.path.join(a, "r2.json")))
        assert r2["started_fresh"]
        print("  ok  existing checkpoint at save path is NOT auto-loaded")

        # explicit warm start + freeze_all (transfer protocol)
        b = os.path.join(d, "b")
        v15.run_simulation(episodes=1, out_dir=b, load_from=os.path.join(a, "ck.pkl"),
                           freeze_all=True, **COMMON)
        rb = json.load(open(os.path.join(b, "results.json")))
        assert not rb["started_fresh"] and rb["loaded_from"].endswith("ck.pkl")
        assert rb["episodes"][0]["mean_b_change"] == r2["episodes"][-1]["mean_b_change"], \
            "freeze_all must not change B"
        print("  ok  --load + --freeze_all: B unchanged during evaluation")

        # const control fed from the full run's measured mean signal
        c = os.path.join(d, "c")
        cc = e["cong_obs_mean_by_floor"]
        v15.run_simulation(episodes=1, out_dir=c, cong_source="const", cong_const=cc, **COMMON)
        rc = json.load(open(os.path.join(c, "results.json")))["episodes"][0]
        assert np.allclose(rc["mean_cong_belief"][3:7], cc[0], atol=0.05)
        print(f"  ok  const control: belief pinned at {np.round(cc, 3).tolist()}")
    print("\nRUNNER TEST PASSED")


if __name__ == "__main__":
    main()
