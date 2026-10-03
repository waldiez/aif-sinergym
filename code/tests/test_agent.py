"""
Agent-level regression tests for aif_agent (no EnergyPlus needed).

A small closed-loop toy plant stands in for the building so the agents see
realistic-looking temperature and floor-power sequences.

  python tests/test_agent.py            # all tests, ~1-2 min on CPU
"""
import importlib.util, os, sys, tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


v14 = load("v14", os.path.join(HERE, "aif_agent_v14_reference.py"))
v15 = load("v15", os.path.join(ROOT, "aif_agent.py"))
N = 15


class Box:  # minimal action_space stand-in
    def __init__(self):
        self.low = np.array([15.0] * N + [22.5] * N, dtype=np.float32)
        self.high = np.array([22.5] * N + [30.0] * N, dtype=np.float32)


class FakeEnv:
    action_space = Box()


class ToyPlant:
    """First-order zones + electric reheat/DX, 3 floors x 5 zones.
    Produces a 95-dim obs with the same index layout as register_env."""
    def __init__(self, seed=0, start_month=1, start_day=28):
        self.rng = np.random.default_rng(seed)
        self.T = 18.0 + self.rng.normal(0, 1.0, N)
        self.k = np.where(np.arange(N) < 3, 0.015, 0.035)
        self.m, self.d, self.h = start_month, start_day, 0.0
        self.floor = np.array([0, 1, 2] + [0] * 4 + [1] * 4 + [2] * 4)

    def outdoor(self):
        return 2.0 + 6.0 * np.sin(2 * np.pi * (self.h - 9) / 24.0)

    def obs(self, fp):
        o = np.zeros(95, dtype=np.float32)
        o[0], o[1], o[2] = self.m, self.d, int(self.h)
        o[3] = self.outdoor(); o[8] = max(0.0, 400 * np.sin(np.pi * (self.h - 6) / 12))
        o[9:24] = self.T
        o[92:95] = fp * 900.0          # meters report J per 15-min step
        return o

    def step(self, action):
        htg, clg = action[:N], action[N:]
        To = self.outdoor()
        heat = np.clip(htg - self.T, 0, 1.5); cool = np.clip(self.T - clg, 0, 1.5)
        occ = 1.0 if 7 <= self.h <= 19 else 0.0
        self.T = (self.T + self.k * (To - self.T) + 0.8 * heat - 0.8 * cool
                  + 0.1 * occ + self.rng.normal(0, 0.05, N))
        zone_kw = 4.0 * heat + 3.0 * cool
        fp = np.array([zone_kw[self.floor == f].sum() for f in range(3)]) * 1000.0  # W
        self.h += 0.25
        if self.h >= 24:
            self.h = 0.0; self.d += 1
            if self.d > 31 or (self.m == 2 and self.d > 28):
                self.d = 1; self.m += 1
        return fp


def run_closed_loop(agent, steps, seed=0, feed_cong=True, scale=0.0011111):
    """Drive a PyMDPOffice-like agent; return the (steps, 30) action array."""
    plant = ToyPlant(seed)
    fp = np.zeros(3)
    acts = []
    for _ in range(steps):
        o = plant.obs(fp)
        if feed_cong:
            agent.observe_floor_power([float(o[i]) * scale for i in (92, 93, 94)])
        a = agent.get_action(o)
        acts.append(a.copy())
        fp = plant.step(a)
    return np.asarray(acts)


FULL = dict(mode="zone", structural=True, energy_weight=0.2, tou_weight=0.0,
            congestion_weight=5.0, action_temp=0.0, policy_len=8, learn_C=True,
            c_lr=0.02, energy_w_min=0.0, energy_w_max=1.0, deadband_weight=8.0,
            device="cpu", dtype="double")


def test_legacy_reproduces_v14(steps=600):
    a14 = run_closed_loop(v14.PyMDPOffice(FakeEnv(), **FULL), steps)
    a15 = run_closed_loop(v15.PyMDPOffice(FakeEnv(), legacy_infer=True,
                                          horizon_ref=0, **FULL), steps)
    assert np.array_equal(a14, a15), "v15 --legacy_infer must reproduce v14"
    a15h = run_closed_loop(v15.PyMDPOffice(FakeEnv(), legacy_infer=True,
                                           horizon_ref=8, **FULL), steps)
    assert np.array_equal(a14, a15h), "horizon_ref == policy_len must be a no-op"
    print(f"  ok  legacy mode == v14 over {steps} closed-loop steps "
          f"(also with horizon_ref=8, policy_len=8)")


def test_infer_fix_effect(steps=600):
    a_old = run_closed_loop(v15.PyMDPOffice(FakeEnv(), legacy_infer=True, **FULL), steps)
    a_new = run_closed_loop(v15.PyMDPOffice(FakeEnv(), **FULL), steps)
    frac = np.mean(np.any(a_old != a_new, axis=1))
    print(f"  info F1 (infer fix) changes the action at {100*frac:.1f}% of steps "
          f"on the toy plant")


def test_posterior_math():
    """F1: the fixed update equals the exact marginal of the joint (T,H)
    posterior; the legacy one does not."""
    ag = v15.PyMDPOffice(FakeEnv(), **FULL).batch
    rng = np.random.default_rng(1)
    T0 = 20 + rng.normal(0, 1, N)
    ag.update_context(1, 15, 10, True, 3.0, 0.0)
    ag.step(T0, True)                               # first step: sets prev_action
    ag.qH = torch.softmax(torch.as_tensor(rng.normal(0, 1, (N, ag.N_H))), 1)
    T1 = T0 + rng.normal(0, 0.5, N)
    qT_prev, qH_prev, qO, qW = ag.qT.clone(), ag.qH.clone(), ag.qO.clone(), ag.qW.clone()
    ag._infer(T1, True)
    # exact joint posterior over (x, h) at the current step
    t_idx = torch.tensor([ag._t2s(T) for T in T1])
    like_T = ag.A_T[t_idx, :]
    ar = torch.arange(N)
    B_a = ag.B_T.permute(0, 6, 1, 2, 3, 4, 5)[ar, ag.prev_action]
    pred = torch.einsum('nowhxt,nt,no,nw->nxh', B_a, qT_prev, qO, qW)
    prior_H = qH_prev @ ag.B_H.T
    joint = like_T[:, :, None] * pred * prior_H[:, None, :]
    qT_exact = joint.sum(2); qT_exact /= qT_exact.sum(1, keepdim=True)
    assert torch.allclose(ag.qT, qT_exact, atol=1e-10), "fixed update != exact marginal"
    print("  ok  fixed temperature update equals the exact joint-posterior marginal")


def test_n1_equivalence(steps=150):
    """Each row of the N=15 batch equals an independent N=1 agent fed the same
    inputs (licenses deploying 15 separate actors from one model)."""
    kw = dict(energy_weight=0.2, tou_weight=0.0, congestion_weight=5.0,
              action_temp=0.0, policy_len=8, learn_C=True, c_lr=0.02,
              energy_w_min=0.0, energy_w_max=1.0, deadband_weight=8.0,
              device="cpu", dtype="double")
    ctxs = [dict(v15.ZONE_META[i]) for i in range(N)]
    big = v15.BatchedFactoredAgents(N, 15.0, 22.5, 22.5, 30.0, ctxs, **kw)
    small = [v15.BatchedFactoredAgents(1, 15.0, 22.5, 22.5, 30.0, [ctxs[i]], **kw)
             for i in range(N)]
    rng = np.random.default_rng(3)
    T = 19 + rng.normal(0, 1, N)
    for t in range(steps):
        hour = (t * 0.25) % 24
        cong = rng.uniform(0, 1, N)
        for ag in [big] + small:
            ag.update_context(1, 16, int(hour), 7 <= hour <= 19, 3.0, 100.0)
        big.observe_congestion(cong)
        for i, ag in enumerate(small):
            ag.observe_congestion(cong[i:i + 1])
        h_b, c_b = big.step(T, 7 <= hour <= 19)
        for i, ag in enumerate(small):
            h_s, c_s = ag.step(T[i:i + 1], 7 <= hour <= 19)
            assert h_s[0] == h_b[i] and c_s[0] == c_b[i], f"agent {i} diverged at t={t}"
        T = T + 0.3 * (h_b > T) - 0.3 * (c_b < T) + rng.normal(0, 0.1, N)
    print(f"  ok  N=15 batch == 15 independent N=1 agents over {steps} steps")


def test_checkpoint_roundtrip(steps=300):
    a = v15.PyMDPOffice(FakeEnv(), **FULL)
    run_closed_loop(a, steps)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ck.pkl"); a.save(p)
        b = v15.PyMDPOffice(FakeEnv(), **FULL); b.load(p)
    assert abs(a._cong_ref - b._cong_ref) < 1e-9, "cong_ref not restored"
    assert torch.equal(a.batch.B_T, b.batch.B_T)
    assert torch.equal(a.batch.energy_w, b.batch.energy_w)
    a.reset(); b.reset()
    xa = run_closed_loop(a, 200, seed=7); xb = run_closed_loop(b, 200, seed=7)
    assert np.array_equal(xa, xb), "restored agent behaves differently"
    print("  ok  checkpoint round-trip restores B, energy_w and cong_ref exactly")


def test_freeze_all(steps=300):
    a = v15.PyMDPOffice(FakeEnv(), freeze_B=True, freeze_C=True, **FULL)
    B0 = a.batch.B_T.clone(); w0 = a.batch.energy_w.clone()
    run_closed_loop(a, steps)
    assert torch.equal(a.batch.B_T, B0) and torch.equal(a.batch.energy_w, w0)
    print("  ok  freeze_all keeps B and energy_w fixed")


def test_reset_clears_filters(steps=200):
    a = v15.PyMDPOffice(FakeEnv(), **FULL)
    run_closed_loop(a, steps)
    assert float(a.batch.cong.abs().sum()) > 0
    w = a.batch.energy_w.clone()
    a.reset()
    assert float(a.batch.cong.abs().sum()) == 0 and float(a.batch.cerr_ema.abs().sum()) == 0
    assert torch.equal(a.batch.energy_w, w), "learned energy_w must persist"
    print("  ok  reset clears congestion/comfort filters, keeps learned energy_w")


def test_cong_sources():
    fp = np.array([30.0, 20.0, 10.0])
    for src, expect in [("others", [30 / 60, 40 / 60, 50 / 60]),
                        ("own", [1.0, 40 / 60, 20 / 60]),
                        ("building", [2 / 3] * 3)]:
        a = v15.PyMDPOffice(FakeEnv(), cong_source=src, cong_ref_W=60.0, **FULL)
        got = a._cong_by_floor(fp)
        assert np.allclose(got, expect), (src, got)
    a = v15.PyMDPOffice(FakeEnv(), cong_source="const", cong_const=[.1, .2, .3], **FULL)
    assert np.allclose(a._cong_by_floor(fp), [.1, .2, .3])
    a = v15.PyMDPOffice(FakeEnv(), cong_source="lag24", cong_ref_W=60.0, **FULL)
    seq = [np.array([x, 0.0, 0.0]) for x in range(1, 200)]
    outs = [a._cong_by_floor(f)[1] for f in seq]      # mid floor sees bottom's load
    assert np.isclose(outs[150], seq[150 - 96][0] / 60.0), "lag24 must be 96 steps late"
    # equal split -> every dynamic source gives the same value
    eq = np.array([20.0, 20.0, 20.0])
    vals = [v15.PyMDPOffice(FakeEnv(), cong_source=s, cong_ref_W=60.0, **FULL)
            ._cong_by_floor(eq) for s in ("others", "own", "building")]
    assert all(np.allclose(v, vals[0]) for v in vals)
    print("  ok  congestion sources (others/own/building/const/lag24) and scaling")


def test_hetero_weights(steps=100):
    ew = np.linspace(0, 1, N)
    a = v15.PyMDPOffice(FakeEnv(), **{**FULL, "learn_C": False}, energy_w_init=ew)
    assert a.batch.use_agent_w
    run_closed_loop(a, steps)
    assert np.allclose(a.batch.energy_w.numpy(), ew), "fixed weights must not move"
    print("  ok  heterogeneous fixed energy weights are used and stay fixed")


def test_override_counter():
    a = v15.PyMDPOffice(FakeEnv(), **FULL)
    o = ToyPlant().obs(np.zeros(3)); o[9:24] = 10.0      # 10 degC: far out of band
    a.get_action(o); a.get_action(o)
    st = a.get_stats()
    assert st["override_rate"] > 0.9, st["override_rate"]
    print(f"  ok  override counter works (rate {st['override_rate']:.2f} at 10 degC)")


if __name__ == "__main__":
    torch.set_num_threads(1)
    for fn in [test_legacy_reproduces_v14, test_posterior_math, test_infer_fix_effect,
               test_n1_equivalence, test_checkpoint_roundtrip, test_freeze_all,
               test_reset_clears_filters, test_cong_sources, test_hetero_weights,
               test_override_counter]:
        print(fn.__name__); fn()
    print("\nALL AGENT TESTS PASSED")
