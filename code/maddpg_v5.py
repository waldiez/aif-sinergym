"""
MADDPG for Sinergym OfficeMedium — 15 Zone Agents + Centralized Critic
(v4 — Resumable Training with Full Checkpoint Support + CS metric)

Changes from previous v4:
  - Added comfort_score (CS) metric, imported from ensemble_controller.
    Tracked in both train() and evaluate(), at monthly and episode level.
  - CS is reported as a weighted average across active periods (max=1.0)
    PLUS a CS_occ diagnostic that uses only fully-occupied steps.
  - Recall: ensemble_controller.compute_comfort_score has been edited so
    it returns the raw mean score (no state_weight multiplication). We
    apply state_weight here at the call site as a proper weighted average.

Architecture & buffer / checkpoint logic unchanged from v4.

Per-agent local observation:
  - Shared global: time features (11) + outdoor weather (6) + HVAC power (2) = 19
  - Zone-specific: own temp (1) + own humidity (1) + own occupancy (1)
                   + own setpoints (2) + zone type encoding (4) + own temp delta (1)
  = 19 + 10 = 29 features per agent

Observation layout (92):
  [0-2]     time: month, day, hour
  [3-8]     outdoor: temp, humidity, wind_speed, wind_direction, diffuse_solar, direct_solar
  [9-26]    zone air temperatures (18 zones)
  [27-44]   zone air humidity (18 zones)
  [45-59]   zone occupancy (15 occupied zones)
  [60-89]   setpoints paired: [htg_z0, clg_z0, htg_z1, clg_z1, ...] (30 values)
  [90]      HVAC electricity demand rate (W)
  [91]      total electricity HVAC meter (J)

Action layout (30):
  [0-14]    heating setpoints per zone  [15.0, 22.0]
  [15-29]   cooling setpoints per zone  [24.0, 30.0]
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from typing import List, Optional, Dict, Any

# Single source of truth for reward, comfort scoring, schedule reward, the
# comfort/deviation metrics, AND the real-world $ cost: everything comes from
# metrics_utils (shared with the AIF and RBC runs). MADDPG trains on the
# CustomRewardWrapper reward from metrics_utils (energy+TOU merged), and the
# wrapper also reports per-step cumulative $ cost in info.
from metrics_utils import (
    CustomRewardWrapper,
    CSAccumulator,
    compute_comfort_score,
    compute_hourly_linear_reward,
    compute_zone_comfort_rate,
    compute_mean_deviation,
    load_factor,
    floor_coincidence_factor,
)


# =============================================================================
# ZONE / OBS CONFIGURATION
# =============================================================================
N_OCCUPIED = 15

COMFORT_WINTER = (20.0, 23.5)
COMFORT_SUMMER = (23.0, 26.0)

IDX_MONTH = 0
IDX_DAY = 1
IDX_HOUR = 2
IDX_OUTDOOR_TEMP = 3
IDX_DIRECT_SOLAR = 8
IDX_ZONE_TEMPS = slice(9, 24)
IDX_ZONE_TEMPS_ALL = slice(9, 27)
IDX_HVAC_DEMAND = 90
IDX_METER = 91

ZONE_META = {
    0:  {"type": "core",      "floor": "bottom", "orient": "none"},
    1:  {"type": "core",      "floor": "mid",    "orient": "none"},
    2:  {"type": "core",      "floor": "top",    "orient": "none"},
    3:  {"type": "perimeter", "floor": "bottom", "orient": "south"},
    4:  {"type": "perimeter", "floor": "bottom", "orient": "east"},
    5:  {"type": "perimeter", "floor": "bottom", "orient": "north"},
    6:  {"type": "perimeter", "floor": "bottom", "orient": "west"},
    7:  {"type": "perimeter", "floor": "mid",    "orient": "south"},
    8:  {"type": "perimeter", "floor": "mid",    "orient": "east"},
    9:  {"type": "perimeter", "floor": "mid",    "orient": "north"},
    10: {"type": "perimeter", "floor": "mid",    "orient": "west"},
    11: {"type": "perimeter", "floor": "top",    "orient": "south"},
    12: {"type": "perimeter", "floor": "top",    "orient": "east"},
    13: {"type": "perimeter", "floor": "top",    "orient": "north"},
    14: {"type": "perimeter", "floor": "top",    "orient": "west"},
}

ORIENT_ANGLES = {"none": 0, "north": 0, "east": 90, "south": 180, "west": 270}
FLOOR_MAP = {"bottom": 0.0, "mid": 0.5, "top": 1.0}

def get_zone_encoding(zone_idx):
    meta = ZONE_META[zone_idx]
    is_core = 1.0 if meta["type"] == "core" else 0.0
    floor_norm = FLOOR_MAP[meta["floor"]]
    angle_deg = ORIENT_ANGLES[meta["orient"]]
    angle_rad = np.radians(angle_deg)
    return np.array([is_core, floor_norm, np.sin(angle_rad), np.cos(angle_rad)], dtype=np.float32)

ZONE_ENCODINGS = np.stack([get_zone_encoding(i) for i in range(N_OCCUPIED)])
ZONE_FLOOR_IDX = np.array([{"bottom": 0, "mid": 1, "top": 2}[ZONE_META[i]["floor"]]
                           for i in range(N_OCCUPIED)])


def enable_floor_features(on=True):
    """v5: must be called BEFORE any agent / normaliser / buffer is built."""
    Config.FLOOR_FEATS = bool(on)
    Config.LOCAL_OBS_DIM = 13 if on else 10
    Config.AGENT_OBS_DIM = Config.GLOBAL_OBS_DIM + Config.LOCAL_OBS_DIM


def get_seasonal_comfort(month, day):
    is_summer = (month > 6 or (month == 6 and day >= 1)) and \
                (month < 10 or (month == 9 and day <= 30))
    return COMFORT_SUMMER if is_summer else COMFORT_WINTER


def get_occupancy_state(month, day, hour):
    try:
        from datetime import datetime
        is_weekend = datetime(2024, int(month), int(day)).weekday() >= 5
    except:
        is_weekend = False
    if is_weekend:
        return 'unoccupied'
    elif 7 <= hour <= 19:
        return 'occupied'
    elif 5 <= hour < 7:
        return 'pre_occupy'
    elif 19 < hour <= 21:
        return 'post_occupy'
    else:
        return 'unoccupied'


# =============================================================================
# METRICS (legacy)
# =============================================================================
def is_occupied_hour(obs):
    month, day, hour = int(obs[IDX_MONTH]), int(obs[IDX_DAY]), int(obs[IDX_HOUR])
    return get_occupancy_state(month, day, hour) == 'occupied'

# compute_zone_comfort_rate and compute_mean_deviation now come from
# metrics_utils (imported above) so all controllers are scored identically.

def count_zones_ok(obs):
    month, day = int(obs[IDX_MONTH]), int(obs[IDX_DAY])
    cl, ch = get_seasonal_comfort(month, day)
    zt = obs[IDX_ZONE_TEMPS]
    return int(np.sum((zt >= cl) & (zt <= ch)))


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    TRAIN_EPISODES = 10
    WARMUP_EPISODES = 1

    SEQ_LEN = 8
    GRU_HIDDEN = 64
    GRU_LAYERS = 1

    BUFFER_SIZE = 500_000
    BATCH_SIZE = 256
    GAMMA = 0.99
    TAU = 0.005
    LR_ACTOR = 1e-4
    LR_CRITIC = 3e-4

    POLICY_DELAY = 2
    TARGET_NOISE_STD = 0.2
    TARGET_NOISE_CLIP = 0.5

    NOISE_STD = 0.10
    NOISE_DECAY = 0.99995
    NOISE_MIN = 0.02

    ACTOR_HIDDEN = 128
    CRITIC_HIDDEN = 256

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    W_ENERGY     = 0.20
    W_COMFORT    = 0.40
    W_ACTION     = 0.00
    W_DEADBAND   = 0.05
    W_TEMP_TARGET= 0.10
    W_PEAK       = 0.17
    W_TOU        = 0.08

    UPDATES_PER_STEP = 2
    GRAD_CLIP_CRITIC = 1.0
    GRAD_CLIP_ACTOR = 0.5

    WEATHER = "mixed"
    # Per-floor coincidence-factor config (game-theoretic metric across the 3
    # air-loop floors). Set FLOOR_POWER_IDX to "bot,mid,top" obs indices (e.g.
    # "92,93,94"); meters report Joules so use FLOOR_POWER_SCALE=0.0011111 for kW.
    FLOOR_POWER_IDX = "92,93,94"      # default matches the AIF/RBC runs
    FLOOR_POWER_SCALE = 0.0011111     # J/timestep -> kW (CF is scale-invariant)
    WARMUP_RBC = "proactive"          # {proactive, reactive} RBC for warm-start

    GLOBAL_OBS_DIM = 19
    LOCAL_OBS_DIM = 10
    AGENT_OBS_DIM = 29
    # v5: per-agent metered floor power (same signals the AIF agents see):
    # [own floor, the two other floors, whole building] in kW.
    FLOOR_FEATS = False

    CHECKPOINT_DIR = "checkpoints"


# =============================================================================
# OBSERVATION BUILDER
# =============================================================================
class ObservationBuilder:
    def __init__(self):
        self.prev_zone_temps = None

    def reset(self):
        self.prev_zone_temps = None

    def get_state(self):
        return {'prev_zone_temps': self.prev_zone_temps.copy() if self.prev_zone_temps is not None else None}

    def set_state(self, state):
        pzt = state.get('prev_zone_temps')
        self.prev_zone_temps = np.array(pzt, dtype=np.float32) if pzt is not None else None

    def build_agent_obs(self, obs_raw):
        month, day, hour = obs_raw[0], obs_raw[1], obs_raw[2]
        day_of_year = (month - 1) * 30.44 + day

        def sc(val, period):
            a = 2 * np.pi * val / period
            return np.sin(a), np.cos(a)

        h_s, h_c = sc(hour, 24)
        d_s, d_c = sc(day, 31)
        m_s, m_c = sc(month - 1, 12)
        y_s, y_c = sc(day_of_year, 365.25)

        try:
            from datetime import datetime
            dow = datetime(2024, int(month), int(day)).weekday()
            is_wknd = 1.0 if dow >= 5 else 0.0
        except:
            is_wknd = 0.0
        is_occ = 1.0 if (is_wknd == 0 and 7 <= hour <= 19) else 0.0
        h_norm = hour / 23.0

        time_feats = np.array([h_s, h_c, d_s, d_c, m_s, m_c, y_s, y_c,
                               is_wknd, is_occ, h_norm], dtype=np.float32)
        weather = obs_raw[3:9].astype(np.float32)
        hvac = np.array([obs_raw[IDX_HVAC_DEMAND], obs_raw[IDX_METER]], dtype=np.float32)
        global_feats = np.concatenate([time_feats, weather, hvac])

        zone_temps = obs_raw[9:24].astype(np.float32)
        zone_humidity = obs_raw[27:42].astype(np.float32)
        zone_occupancy = obs_raw[45:60].astype(np.float32)

        htg_setpoints = np.array([obs_raw[60 + i * 2] for i in range(N_OCCUPIED)], dtype=np.float32)
        clg_setpoints = np.array([obs_raw[60 + i * 2 + 1] for i in range(N_OCCUPIED)], dtype=np.float32)

        if self.prev_zone_temps is not None:
            temp_deltas = zone_temps - self.prev_zone_temps
        else:
            temp_deltas = np.zeros(N_OCCUPIED, dtype=np.float32)
        self.prev_zone_temps = zone_temps.copy()

        if Config.FLOOR_FEATS:
            fpi = [int(x) for x in str(Config.FLOOR_POWER_IDX).split(",")]
            fp_kw = np.array([float(obs_raw[j]) * Config.FLOOR_POWER_SCALE / 1000.0
                              for j in fpi], dtype=np.float32)
            fp_tot = float(fp_kw.sum())

        agent_obs = np.zeros((N_OCCUPIED, Config.AGENT_OBS_DIM), dtype=np.float32)
        for i in range(N_OCCUPIED):
            local = [
                zone_temps[i], zone_humidity[i], zone_occupancy[i],
                htg_setpoints[i], clg_setpoints[i],
                ZONE_ENCODINGS[i, 0], ZONE_ENCODINGS[i, 1],
                ZONE_ENCODINGS[i, 2], ZONE_ENCODINGS[i, 3],
                temp_deltas[i],
            ]
            if Config.FLOOR_FEATS:
                own = float(fp_kw[ZONE_FLOOR_IDX[i]])
                local += [own, fp_tot - own, fp_tot]
            agent_obs[i] = np.concatenate([global_feats, np.array(local, dtype=np.float32)])

        return agent_obs


# =============================================================================
# CUSTOM REWARD WRAPPER + CS METRIC  ->  imported from metrics_utils
# =============================================================================
# CustomRewardWrapper and CSAccumulator now live in metrics_utils (single
# source of truth, shared with the AIF and RBC runs). The metrics_utils
# wrapper merges energy+TOU into one term and reports real-world $ cost in
# info (cost_energy_usd / cost_demand_usd / cost_total_usd). They are imported
# at the top of this file; the previous inline copies have been removed.


# =============================================================================
# NORMALIZER
# =============================================================================
class AgentObsNormalizer:
    def __init__(self, obs_dim):
        self.obs_dim = obs_dim
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.var = np.ones(obs_dim, dtype=np.float64)
        self.count = 0
        self.frozen = False

    def freeze(self):
        self.frozen = True
        self.var = np.maximum(self.var, 1e-4)
        std = np.sqrt(self.var / max(1, self.count - 1))
        print(f"  Agent normalizer frozen at count={self.count}")
        print(f"    Std range: [{std.min():.4f}, {std.max():.4f}]")

    def update(self, agent_obs_all):
        if self.frozen:
            return
        for obs in agent_obs_all:
            self.count += 1
            obs64 = obs.astype(np.float64)
            if self.count == 1:
                self.mean = obs64.copy()
                self.var = np.zeros_like(self.mean)
            else:
                delta = obs64 - self.mean
                self.mean += delta / self.count
                delta2 = obs64 - self.mean
                self.var += delta * delta2

    def normalize(self, obs):
        if self.count < 2:
            return obs.astype(np.float32)
        std = np.sqrt(np.maximum(self.var / max(1, self.count - 1), 1e-4)) + 1e-8
        return ((obs.astype(np.float64) - self.mean) / std).astype(np.float32)

    def save(self, path):
        np.savez(path, mean=self.mean, var=self.var, count=self.count, frozen=self.frozen)

    def load(self, path):
        d = np.load(path)
        self.mean, self.var = d['mean'], d['var']
        self.count, self.frozen = int(d['count']), bool(d['frozen'])


# =============================================================================
# REPLAY BUFFER WITH SAVE/LOAD
# =============================================================================
class MADDPGReplayBuffer:
    def __init__(self, capacity, n_agents, seq_len, agent_obs_dim, action_per_agent=2):
        self.capacity = capacity
        self.n_agents = n_agents
        self.seq_len = seq_len
        self.agent_obs_dim = agent_obs_dim
        self.action_per_agent = action_per_agent
        total_action = n_agents * action_per_agent

        self.states = np.zeros((capacity, n_agents, seq_len, agent_obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, total_action), dtype=np.float32)
        self.global_rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.zone_rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_states = np.zeros((capacity, n_agents, seq_len, agent_obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def push(self, state, action, global_reward, zone_rewards, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.global_rewards[self.ptr, 0] = global_reward
        self.zone_rewards[self.ptr] = zone_rewards
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr, 0] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, normalizer=None):
        idxs = np.random.randint(0, self.size, size=batch_size)

        states = self.states[idxs].copy()
        next_states = self.next_states[idxs].copy()
        actions = self.actions[idxs].copy()
        g_rewards = self.global_rewards[idxs].copy()
        z_rewards = self.zone_rewards[idxs].copy()
        dones = self.dones[idxs].copy()

        if normalizer is not None:
            B, N, T, D = states.shape
            flat = states.reshape(-1, D)
            flat = normalizer.normalize(flat)
            states = flat.reshape(B, N, T, D)

            flat_n = next_states.reshape(-1, D)
            flat_n = normalizer.normalize(flat_n)
            next_states = flat_n.reshape(B, N, T, D)

        dev = Config.DEVICE
        return (
            torch.from_numpy(states).to(dev),
            torch.from_numpy(actions).to(dev),
            torch.from_numpy(g_rewards).to(dev),
            torch.from_numpy(z_rewards).to(dev),
            torch.from_numpy(next_states).to(dev),
            torch.from_numpy(dones).to(dev),
        )

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        meta = {'ptr': self.ptr, 'size': self.size,
                'capacity': self.capacity, 'n_agents': self.n_agents,
                'seq_len': self.seq_len, 'agent_obs_dim': self.agent_obs_dim,
                'action_per_agent': self.action_per_agent}
        with open(os.path.join(directory, 'buffer_meta.json'), 'w') as f:
            json.dump(meta, f)

        n = self.size
        np.save(os.path.join(directory, 'states.npy'), self.states[:n])
        np.save(os.path.join(directory, 'actions.npy'), self.actions[:n])
        np.save(os.path.join(directory, 'global_rewards.npy'), self.global_rewards[:n])
        np.save(os.path.join(directory, 'zone_rewards.npy'), self.zone_rewards[:n])
        np.save(os.path.join(directory, 'next_states.npy'), self.next_states[:n])
        np.save(os.path.join(directory, 'dones.npy'), self.dones[:n])
        print(f"  Buffer saved: {n} transitions to {directory}/")

    def load(self, directory):
        meta_path = os.path.join(directory, 'buffer_meta.json')
        if not os.path.exists(meta_path):
            print(f"  Warning: No buffer found at {directory}")
            return False

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.ptr = meta['ptr']
        self.size = meta['size']
        n = self.size

        self.states[:n] = np.load(os.path.join(directory, 'states.npy'))
        self.actions[:n] = np.load(os.path.join(directory, 'actions.npy'))
        self.global_rewards[:n] = np.load(os.path.join(directory, 'global_rewards.npy'))
        self.zone_rewards[:n] = np.load(os.path.join(directory, 'zone_rewards.npy'))
        self.next_states[:n] = np.load(os.path.join(directory, 'next_states.npy'))
        self.dones[:n] = np.load(os.path.join(directory, 'dones.npy'))
        print(f"  Buffer loaded: {n} transitions from {directory}/")
        return True

    def __len__(self):
        return self.size


class AgentSequenceBuilder:
    def __init__(self, n_agents, seq_len, obs_dim):
        self.n_agents = n_agents
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.buffers = np.zeros((n_agents, seq_len, obs_dim), dtype=np.float32)

    def reset(self, first_obs):
        self.buffers = np.zeros((self.n_agents, self.seq_len, self.obs_dim), dtype=np.float32)
        for i in range(self.n_agents):
            self.buffers[i, -1] = first_obs[i]

    def append(self, obs):
        self.buffers = np.roll(self.buffers, -1, axis=1)
        for i in range(self.n_agents):
            self.buffers[i, -1] = obs[i]
        return self.buffers.copy()

    def get_current(self):
        return self.buffers.copy()


# =============================================================================
# NETWORKS
# =============================================================================
class ZoneGRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x.contiguous()
        h0 = torch.zeros(1, x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        output, _ = self.gru(x, h0)
        return self.ln(output[:, -1, :])


class ZoneActor(nn.Module):
    def __init__(self, obs_dim, gru_hidden=64, mlp_hidden=128):
        super().__init__()
        self.encoder = ZoneGRUEncoder(obs_dim, gru_hidden)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        layers = [m for m in self.head if isinstance(m, nn.Linear)]
        nn.init.uniform_(layers[-1].weight, -3e-3, 3e-3)
        nn.init.constant_(layers[-1].bias, 0)

    def forward(self, seq):
        h = self.encoder(seq)
        return (torch.tanh(self.head(h)) + 1.0) / 2.0


class CentralizedCritic(nn.Module):
    def __init__(self, agent_obs_dim, n_agents, total_action_dim,
                 gru_hidden=64, mlp_hidden=256):
        super().__init__()
        self.n_agents = n_agents
        self.shared_encoder = ZoneGRUEncoder(agent_obs_dim, gru_hidden)

        critic_input_dim = n_agents * gru_hidden + total_action_dim
        self.head = nn.Sequential(
            nn.Linear(critic_input_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        layers = [m for m in self.head if isinstance(m, nn.Linear)]
        nn.init.uniform_(layers[-1].weight, -3e-3, 3e-3)

    def forward(self, agent_seqs, joint_action):
        B = agent_seqs.size(0)
        encodings = []
        for i in range(self.n_agents):
            enc = self.shared_encoder(agent_seqs[:, i])
            encodings.append(enc)
        all_enc = torch.cat(encodings, dim=1)
        return self.head(torch.cat([all_enc, joint_action], dim=1))


class TwinCentralizedCritic(nn.Module):
    def __init__(self, agent_obs_dim, n_agents, total_action_dim,
                 gru_hidden=64, mlp_hidden=256):
        super().__init__()
        self.q1 = CentralizedCritic(agent_obs_dim, n_agents, total_action_dim,
                                     gru_hidden, mlp_hidden)
        self.q2 = CentralizedCritic(agent_obs_dim, n_agents, total_action_dim,
                                     gru_hidden, mlp_hidden)

    def forward(self, agent_seqs, joint_action):
        return self.q1(agent_seqs, joint_action), self.q2(agent_seqs, joint_action)

    def q1_forward(self, agent_seqs, joint_action):
        return self.q1(agent_seqs, joint_action)


# =============================================================================
# MADDPG AGENT
# =============================================================================
class MADDPGAgent:
    def __init__(self, env):
        self.n_agents = N_OCCUPIED
        self.agent_obs_dim = Config.AGENT_OBS_DIM
        self.action_per_agent = 2
        self.total_action_dim = self.n_agents * self.action_per_agent

        self.htg_low = float(env.action_space.low[0])
        self.htg_high = float(env.action_space.high[0])
        self.clg_low = float(env.action_space.low[N_OCCUPIED])
        self.clg_high = float(env.action_space.high[N_OCCUPIED])

        self.action_low_np = env.action_space.low.copy()
        self.action_high_np = env.action_space.high.copy()

        dev = Config.DEVICE

        self.actors = [ZoneActor(self.agent_obs_dim, Config.GRU_HIDDEN, Config.ACTOR_HIDDEN).to(dev)
                       for _ in range(self.n_agents)]
        self.actors_target = [ZoneActor(self.agent_obs_dim, Config.GRU_HIDDEN, Config.ACTOR_HIDDEN).to(dev)
                              for _ in range(self.n_agents)]
        for i in range(self.n_agents):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())

        self.actor_opts = [optim.Adam(a.parameters(), lr=Config.LR_ACTOR) for a in self.actors]

        self.critic = TwinCentralizedCritic(
            self.agent_obs_dim, self.n_agents, self.total_action_dim,
            Config.GRU_HIDDEN, Config.CRITIC_HIDDEN
        ).to(dev)
        self.critic_target = TwinCentralizedCritic(
            self.agent_obs_dim, self.n_agents, self.total_action_dim,
            Config.GRU_HIDDEN, Config.CRITIC_HIDDEN
        ).to(dev)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=Config.LR_CRITIC)

        self.buffer = MADDPGReplayBuffer(
            Config.BUFFER_SIZE, self.n_agents, Config.SEQ_LEN,
            self.agent_obs_dim, self.action_per_agent
        )

        self.noise_std = Config.NOISE_STD
        self.train_steps = 0
        self.critic_updates = 0

        actor_params = sum(p.numel() for a in self.actors for p in a.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        print(f"  Zone Actors: {self.n_agents} × {sum(p.numel() for p in self.actors[0].parameters()):,} "
              f"= {actor_params:,} total")
        print(f"  Twin Critic: {critic_params:,}")
        print(f"  Grand total: {actor_params + critic_params:,}")

    def select_action(self, agent_seqs, normalizer, add_noise=True):
        dev = Config.DEVICE
        action = np.zeros(self.total_action_dim, dtype=np.float32)

        for i, actor in enumerate(self.actors):
            actor.eval()
            seq_norm = normalizer.normalize(agent_seqs[i])
            seq_t = torch.FloatTensor(seq_norm).unsqueeze(0).to(dev)
            with torch.no_grad():
                a01 = actor(seq_t).cpu().numpy()[0]
            actor.train()

            htg = self.htg_low + a01[0] * (self.htg_high - self.htg_low)
            clg = self.clg_low + a01[1] * (self.clg_high - self.clg_low)

            if add_noise:
                htg += np.random.normal(0, self.noise_std) * (self.htg_high - self.htg_low)
                clg += np.random.normal(0, self.noise_std) * (self.clg_high - self.clg_low)

            action[i] = np.clip(htg, self.htg_low, self.htg_high)
            action[N_OCCUPIED + i] = np.clip(clg, self.clg_low, self.clg_high)

        if add_noise:
            self.noise_std = max(Config.NOISE_MIN, self.noise_std * Config.NOISE_DECAY)

        return action

    def _actions_from_actors(self, actors, agent_seqs):
        B = agent_seqs.size(0)
        htg_range = self.htg_high - self.htg_low
        clg_range = self.clg_high - self.clg_low

        htg_list = []
        clg_list = []
        for i, actor in enumerate(actors):
            a01 = actor(agent_seqs[:, i])
            htg = self.htg_low + a01[:, 0:1] * htg_range
            clg = self.clg_low + a01[:, 1:2] * clg_range
            htg_list.append(htg)
            clg_list.append(clg)

        return torch.cat(htg_list + clg_list, dim=1)

    def update(self, normalizer):
        if len(self.buffer) < Config.BATCH_SIZE:
            return 0.0, 0.0

        for a in self.actors:
            a.train()
        self.critic.train()

        states, actions, g_rewards, z_rewards, next_states, dones = \
            self.buffer.sample(Config.BATCH_SIZE, normalizer)

        with torch.no_grad():
            next_actions = self._actions_from_actors(self.actors_target, next_states)
            noise = (torch.randn_like(next_actions) * Config.TARGET_NOISE_STD)
            noise = noise.clamp(-Config.TARGET_NOISE_CLIP, Config.TARGET_NOISE_CLIP)
            action_range = torch.cat([
                torch.full((next_actions.size(0), N_OCCUPIED), self.htg_high - self.htg_low),
                torch.full((next_actions.size(0), N_OCCUPIED), self.clg_high - self.clg_low),
            ], dim=1).to(Config.DEVICE)
            next_actions = next_actions + noise * action_range
            low = torch.FloatTensor(self.action_low_np).to(Config.DEVICE)
            high = torch.FloatTensor(self.action_high_np).to(Config.DEVICE)
            next_actions = next_actions.clamp(low, high)

            tq1, tq2 = self.critic_target(next_states, next_actions)
            target_q = g_rewards + Config.GAMMA * (1 - dones) * torch.min(tq1, tq2)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), Config.GRAD_CLIP_CRITIC)
        self.critic_opt.step()
        self.critic_updates += 1

        actor_loss_val = 0.0
        if self.critic_updates % Config.POLICY_DELAY == 0:
            pred_actions = self._actions_from_actors(self.actors, states)
            actor_loss = -self.critic.q1_forward(states, pred_actions).mean()

            for opt in self.actor_opts:
                opt.zero_grad()
            actor_loss.backward()
            for i in range(self.n_agents):
                nn.utils.clip_grad_norm_(self.actors[i].parameters(), Config.GRAD_CLIP_ACTOR)
                self.actor_opts[i].step()

            for i in range(self.n_agents):
                self._soft_update(self.actors[i], self.actors_target[i])
            self._soft_update(self.critic, self.critic_target)
            actor_loss_val = actor_loss.item()

        self.train_steps += 1
        return critic_loss.item(), actor_loss_val

    def _soft_update(self, src, tgt):
        for tp, sp in zip(tgt.parameters(), src.parameters()):
            tp.data.copy_(Config.TAU * sp.data + (1 - Config.TAU) * tp.data)

    def save(self, path):
        data = {
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'noise_std': self.noise_std,
            'train_steps': self.train_steps,
            'critic_updates': self.critic_updates,
        }
        for i in range(self.n_agents):
            data[f'actor_{i}'] = self.actors[i].state_dict()
            data[f'actor_target_{i}'] = self.actors_target[i].state_dict()
            data[f'actor_opt_{i}'] = self.actor_opts[i].state_dict()
        torch.save(data, path)
        print(f"  Model saved: {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=Config.DEVICE)
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self.noise_std = ckpt['noise_std']
        self.train_steps = ckpt['train_steps']
        self.critic_updates = ckpt['critic_updates']
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(ckpt[f'actor_{i}'])
            self.actors_target[i].load_state_dict(ckpt[f'actor_target_{i}'])
            self.actor_opts[i].load_state_dict(ckpt[f'actor_opt_{i}'])
        print(f"  Model loaded: {path}")


# =============================================================================
# FULL CHECKPOINT
# =============================================================================
def save_checkpoint(agent, normalizer, obs_builder, episode, metrics_history,
                    checkpoint_dir=None):
    checkpoint_dir = checkpoint_dir or Config.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_path = os.path.join(checkpoint_dir, "model.pt")
    agent.save(model_path)

    buffer_dir = os.path.join(checkpoint_dir, "buffer")
    agent.buffer.save(buffer_dir)

    norm_path = os.path.join(checkpoint_dir, "normalizer.npz")
    normalizer.save(norm_path)

    metadata = {
        'episode': episode,
        'noise_std': agent.noise_std,
        'train_steps': agent.train_steps,
        'critic_updates': agent.critic_updates,
        'buffer_size': len(agent.buffer),
        'normalizer_frozen': normalizer.frozen,
        'normalizer_count': normalizer.count,
        'obs_builder_prev_zone_temps': (
            obs_builder.prev_zone_temps.tolist()
            if obs_builder.prev_zone_temps is not None else None
        ),
        'metrics_history': {
            'rewards':   [float(x) for x in metrics_history.get('rewards', [])],
            'energy':    [float(x) for x in metrics_history.get('energy', [])],
            'zcr':       [float(x) for x in metrics_history.get('zcr', [])],
            'deviation': [float(x) for x in metrics_history.get('deviation', [])],
            # NEW: persist CS history across runs
            'cs':        [float(x) for x in metrics_history.get('cs', [])],
            'cs_occ':    [float(x) for x in metrics_history.get('cs_occ', [])],
        },
        'config_snapshot': {
            'TRAIN_EPISODES': Config.TRAIN_EPISODES,
            'WARMUP_EPISODES': Config.WARMUP_EPISODES,
            'WEATHER': Config.WEATHER,
            'BUFFER_SIZE': Config.BUFFER_SIZE,
            'BATCH_SIZE': Config.BATCH_SIZE,
            'LR_ACTOR': Config.LR_ACTOR,
            'LR_CRITIC': Config.LR_CRITIC,
            # v5 provenance: which run produced this checkpoint
            'WARMUP_RBC': Config.WARMUP_RBC,
            'FLOOR_FEATS': Config.FLOOR_FEATS,
            'AGENT_OBS_DIM': Config.AGENT_OBS_DIM,
        }
    }
    meta_path = os.path.join(checkpoint_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Full checkpoint saved to {checkpoint_dir}/ (ep {episode+1}, "
          f"buf={len(agent.buffer)}, steps={agent.train_steps})")


def load_checkpoint(agent, normalizer, obs_builder, checkpoint_dir=None):
    checkpoint_dir = checkpoint_dir or Config.CHECKPOINT_DIR
    meta_path = os.path.join(checkpoint_dir, "metadata.json")

    if not os.path.exists(meta_path):
        print(f"  No checkpoint found at {checkpoint_dir}/")
        return 0, {'rewards': [], 'energy': [], 'zcr': [],
                   'deviation': [], 'cs': [], 'cs_occ': []}, False

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    last_episode = metadata['episode']
    print(f"\n  Found checkpoint: episode {last_episode+1}, "
          f"buffer={metadata['buffer_size']}, steps={metadata['train_steps']}")

    saved_cfg = metadata.get('config_snapshot', {})
    if saved_cfg.get('WEATHER') and saved_cfg['WEATHER'] != Config.WEATHER:
        print(f"  ⚠ Warning: Weather changed: {saved_cfg['WEATHER']} → {Config.WEATHER}")

    model_path = os.path.join(checkpoint_dir, "model.pt")
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print("  ⚠ Warning: model.pt not found in checkpoint")

    buffer_dir = os.path.join(checkpoint_dir, "buffer")
    agent.buffer.load(buffer_dir)

    norm_path = os.path.join(checkpoint_dir, "normalizer.npz")
    if os.path.exists(norm_path):
        normalizer.load(norm_path)
        if normalizer.frozen:
            print(f"  Normalizer: frozen (count={normalizer.count})")
        else:
            print(f"  Normalizer: active (count={normalizer.count})")

    pzt = metadata.get('obs_builder_prev_zone_temps')
    obs_builder.set_state({'prev_zone_temps': pzt})

    # Default metrics history with new CS keys, fall back gracefully if older
    # checkpoint didn't have them.
    mh_default = {'rewards': [], 'energy': [], 'zcr': [],
                  'deviation': [], 'cs': [], 'cs_occ': []}
    saved_mh = metadata.get('metrics_history', mh_default)
    for k, default in mh_default.items():
        saved_mh.setdefault(k, default)
    metrics_history = saved_mh

    start_episode = last_episode + 1
    print(f"  ✓ Checkpoint loaded. Resuming from episode {start_episode + 1}")

    return start_episode, metrics_history, True


# =============================================================================
# TRAINING (resumable)
# =============================================================================

def _resolve_floor_idx():
    """Parse Config.FLOOR_POWER_IDX into a 3-int list, or None if unset/invalid."""
    fpi = Config.FLOOR_POWER_IDX
    if isinstance(fpi, str):
        fpi = [int(x) for x in fpi.split(",") if x.strip()]
    if fpi is not None and len(fpi) != 3:
        print(f"[coincidence] WARNING: expected 3 floor-power indices, got {fpi}; "
              f"disabling coincidence metric.")
        fpi = None
    return fpi


def _print_demand_coincidence(power_list, floor_power_rows, floor_idx, indent="    "):
    """Print Peak Demand / Load Factor / Floor coincidence block (shared format)."""
    peak_kW = (max(power_list) / 1000.0) if power_list else float('nan')
    lf = load_factor(power_list)
    print(f"{indent}Peak Demand:          {peak_kW:,.1f} kW")
    print(f"{indent}Load Factor:          {lf:.3f}   (avg/peak; higher=flatter=cheaper)")
    if floor_idx is not None and len(floor_power_rows) > 0:
        fp = np.asarray(floor_power_rows, dtype=np.float64)
        cf = floor_coincidence_factor(fp)
        fpeaks = fp.max(axis=0) / 1000.0
        coinc = fp.sum(axis=1).max() / 1000.0
        print(f"{indent}Floor Peaks (kW):     bot={fpeaks[0]:.1f}  mid={fpeaks[1]:.1f}  "
              f"top={fpeaks[2]:.1f}  (sum individual={fpeaks.sum():.1f})")
        print(f"{indent}Coincident Peak:      {coinc:.1f} kW")
        print(f"{indent}Coincidence Factor:   {cf:.3f}   "
              f"(1=floors peak together/expensive, lower=staggered/cheaper)")


def train(resume=True, checkpoint_dir=None, total_episodes=None,
          warmup_episodes=None):
    """
    Train MADDPG.

    Args:
        warmup_episodes: override Config.WARMUP_EPISODES. Set to 0 to skip
            the RBC warmup entirely and train MADDPG from scratch on its own
            random-noise exploration. When 0, the normalizer is bootstrapped
            from MADDPG's own first episode (not RBC), then frozen at the
            start of episode 2.
    """
    from register_env import make_custom_env

    checkpoint_dir = checkpoint_dir or Config.CHECKPOINT_DIR
    if total_episodes is not None:
        Config.TRAIN_EPISODES = total_episodes
    if warmup_episodes is not None:
        Config.WARMUP_EPISODES = max(0, int(warmup_episodes))

    # Only need RBC if we're actually going to use it
    rbc = None
    _rbc_name = Config.WARMUP_RBC
    if Config.WARMUP_EPISODES > 0:
        if Config.WARMUP_RBC == "reactive":
            from rbc_office_reactive import RuleBasedController as RBC
        else:
            from proactive_rbc import RuleBasedController as RBC

    print("=" * 80)
    print(f"MADDPG TRAINING — {Config.TRAIN_EPISODES} episodes, {N_OCCUPIED} zone agents")
    print(f"  Per-agent: GRU({Config.GRU_HIDDEN}) → MLP({Config.ACTOR_HIDDEN}) → 2 actions")
    print(f"  Critic: centralized, shared GRU encoder, MLP({Config.CRITIC_HIDDEN})")
    print(f"  Sequence: {Config.SEQ_LEN} steps = {Config.SEQ_LEN * 15} min")
    print(f"  Agent obs: {Config.AGENT_OBS_DIM} (global={Config.GLOBAL_OBS_DIM} + local={Config.LOCAL_OBS_DIM})")
    if Config.WARMUP_EPISODES > 0:
        print(f"  Warmup: {Config.WARMUP_EPISODES} ep ({_rbc_name} RBC), "
              f"Learning: {Config.TRAIN_EPISODES - Config.WARMUP_EPISODES} ep")
    else:
        print(f"  Warmup: DISABLED — MADDPG explores from step 0")
        print(f"  Learning: {Config.TRAIN_EPISODES} ep")
    print(f"  Updates/step: {Config.UPDATES_PER_STEP}")
    print(f"  Device: {Config.DEVICE}, Weather: {Config.WEATHER}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print("=" * 80)

    env = make_custom_env(weather=Config.WEATHER)
    env = CustomRewardWrapper(env)
    if Config.WARMUP_EPISODES > 0:
        rbc = RBC(env)

    agent = MADDPGAgent(env)
    normalizer = AgentObsNormalizer(Config.AGENT_OBS_DIM)
    obs_builder = ObservationBuilder()

    start_episode = 0
    metrics_history = {'rewards': [], 'energy': [], 'zcr': [],
                       'deviation': [], 'cs': [], 'cs_occ': []}

    # Normalizer policy:
    #   - Always collect for at least one full episode of observations
    #     before freezing, even when there's no RBC warmup. Freezing before
    #     the normalizer has seen any data would leave count<2 and short-
    #     circuit normalize() back to raw obs for the whole run.
    #   - The boundary is max(1, WARMUP_EPISODES): freeze starting at this
    #     episode. With WARMUP=1 this matches the old behaviour. With
    #     WARMUP=0 it means MADDPG explores raw obs during episode 1, then
    #     normalised obs from episode 2 onward.
    freeze_at_episode = max(1, Config.WARMUP_EPISODES)

    if resume:
        start_episode, metrics_history, loaded = load_checkpoint(
            agent, normalizer, obs_builder, checkpoint_dir
        )
        if loaded:
            if start_episode >= freeze_at_episode and not normalizer.frozen:
                normalizer.freeze()
                print(">>> Normalizer frozen (resumed past warmup) <<<")
        else:
            print("  Starting fresh training run.")

    if start_episode >= Config.TRAIN_EPISODES:
        print(f"\n  Already completed {start_episode} episodes "
              f"(target: {Config.TRAIN_EPISODES}). Nothing to do.")
        print(f"  Increase Config.TRAIN_EPISODES or pass total_episodes= to train more.")
        return agent, normalizer

    all_rewards = metrics_history['rewards']
    all_energy  = metrics_history['energy']
    all_zcr     = metrics_history['zcr']
    all_dev     = metrics_history['deviation']
    all_cs      = metrics_history['cs']
    all_cs_occ  = metrics_history['cs_occ']

    seq_builder = AgentSequenceBuilder(N_OCCUPIED, Config.SEQ_LEN, Config.AGENT_OBS_DIM)

    for ep in range(start_episode, Config.TRAIN_EPISODES):
        is_warmup = ep < Config.WARMUP_EPISODES
        phase = "WARMUP (RBC)" if is_warmup else "LEARNING"

        if ep == freeze_at_episode and not normalizer.frozen:
            normalizer.freeze()
            print(">>> Normalizer frozen <<<")

        obs_raw, info = env.reset()
        obs_builder.reset()
        agent_obs = obs_builder.build_agent_obs(obs_raw)

        if not normalizer.frozen:
            normalizer.update(agent_obs)

        seq_builder.reset(agent_obs)
        current_seqs = seq_builder.get_current()

        ep_reward = 0.0
        ep_orig_reward = 0.0
        ep_energy = 0.0
        power_list = []
        floor_power_rows = []
        _floor_idx = _resolve_floor_idx()
        if _floor_idx is not None and max(_floor_idx) >= len(obs_raw):
            print(f"[coincidence] WARNING: floor_power_idx {_floor_idx} exceeds obs "
                  f"size {len(obs_raw)}; disabling coincidence metric.")
            _floor_idx = None
        monthly_cost_energy = 0.0; monthly_cost_demand = 0.0
        prev_ce = 0.0; prev_cd = 0.0
        steps = 0
        terminated = truncated = False
        current_month = 0

        total_zones_ok = 0
        total_zone_checks = 0
        total_deviation = 0.0
        total_dev_steps = 0
        monthly_reward = 0.0
        monthly_zones_ok = 0
        monthly_zone_checks = 0
        monthly_deviation = 0.0
        monthly_dev_steps = 0

        # CS accumulators (episode + monthly)
        cs_ep = CSAccumulator()
        cs_m  = CSAccumulator()

        # HourlyLinearReward (schedule reward) accumulator — reporting only
        ep_hlr = 0.0

        c_losses, a_losses = [], []
        db_violations = 0
        db_total = 0

        print(f"\n{'='*80}")
        print(f"EPISODE {ep+1}/{Config.TRAIN_EPISODES} (Year {ep+1}) — {phase}")
        print("-" * 80)

        while not (terminated or truncated):
            if is_warmup:
                action = rbc.get_action(obs_raw, info)
            else:
                action = agent.select_action(current_seqs, normalizer, add_noise=True)

            next_obs_raw, reward, terminated, truncated, info = env.step(action)
            next_agent_obs = obs_builder.build_agent_obs(next_obs_raw)

            if not normalizer.frozen:
                normalizer.update(next_agent_obs)

            next_seqs = seq_builder.append(next_agent_obs)

            zone_rewards = info.get('zone_rewards', np.zeros(N_OCCUPIED, dtype=np.float32))
            done = 1.0 if terminated else 0.0
            agent.buffer.push(current_seqs, action, reward, zone_rewards, next_seqs, done)

            if not is_warmup:
                for _ in range(Config.UPDATES_PER_STEP):
                    cl, al = agent.update(normalizer)
                    if cl > 0: c_losses.append(cl)
                    if al != 0: a_losses.append(al)

            for i in range(N_OCCUPIED):
                db_total += 1
                if action[i] >= action[N_OCCUPIED + i] - 2.0:
                    db_violations += 1

            ep_reward += reward
            ep_orig_reward += info.get('original_reward', 0)
            ep_energy += info.get('power_W', 0)
            power_list.append(info.get('power_W', 0))
            if _floor_idx is not None:
                floor_power_rows.append([float(next_obs_raw[i]) * Config.FLOOR_POWER_SCALE
                                         for i in _floor_idx])
            ce = info.get('cost_energy_usd', 0.0); cd = info.get('cost_demand_usd', 0.0)
            monthly_cost_energy += ce - prev_ce; prev_ce = ce
            monthly_cost_demand += cd - prev_cd; prev_cd = cd

            # CS update (both granularities)
            cs_ep.update(next_obs_raw)
            cs_m.update(next_obs_raw)

            # HourlyLinearReward (schedule reward) update — reporting only
            hlr, _ = compute_hourly_linear_reward(next_obs_raw)
            ep_hlr += hlr

            zok, ztotal = compute_zone_comfort_rate(next_obs_raw, occupied_only=True)
            total_zones_ok += zok
            total_zone_checks += ztotal
            monthly_zones_ok += zok
            monthly_zone_checks += ztotal

            dev_val, dev_count = compute_mean_deviation(next_obs_raw, occupied_only=True)
            total_deviation += dev_val
            total_dev_steps += dev_count
            monthly_deviation += dev_val
            monthly_dev_steps += dev_count

            steps += 1
            monthly_reward += reward

            m_now = int(next_obs_raw[IDX_MONTH])
            if m_now != current_month:
                if current_month > 0:
                    avg_temp = float(np.mean(next_obs_raw[IDX_ZONE_TEMPS]))
                    zt = next_obs_raw[IDX_ZONE_TEMPS]
                    season = "S" if get_seasonal_comfort(current_month, 15) == COMFORT_SUMMER else "W"
                    m_zcr = (monthly_zones_ok / monthly_zone_checks * 100) if monthly_zone_checks > 0 else 100.0
                    m_dev = monthly_deviation / monthly_dev_steps if monthly_dev_steps > 0 else 0.0
                    m_cs   = cs_m.mean
                    m_cso  = cs_m.mean_occ
                    m_cost = monthly_cost_energy + monthly_cost_demand

                    if is_warmup:
                        print(f"  Month {current_month:2d} [{season}] | "
                              f"R: {monthly_reward:8.2f} | Buf: {len(agent.buffer):6d} | "
                              f"Cost: ${m_cost:7,.0f} | "
                              f"T: {avg_temp:5.1f}°C [{np.min(zt):.1f}-{np.max(zt):.1f}] | "
                              f"CS: {m_cs:.3f} (occ:{m_cso:.3f}) | "
                              f"ZCR: {m_zcr:5.1f}% | Dev: {m_dev:.3f}°C | RBC")
                    else:
                        avg_cl = np.mean(c_losses[-500:]) if c_losses else 0
                        avg_al = np.mean(a_losses[-200:]) if a_losses else 0
                        print(f"  Month {current_month:2d} [{season}] | "
                              f"R: {monthly_reward:8.2f} | C: {avg_cl:.4f} A: {avg_al:.4f} | "
                              f"σ: {agent.noise_std:.4f} | "
                              f"Cost: ${m_cost:7,.0f} | "
                              f"T: {avg_temp:5.1f}°C [{np.min(zt):.1f}-{np.max(zt):.1f}] | "
                              f"CS: {m_cs:.3f} (occ:{m_cso:.3f}) | "
                              f"ZCR: {m_zcr:5.1f}% | Dev: {m_dev:.3f}°C")

                current_month = m_now
                monthly_reward = 0.0
                monthly_cost_energy = 0.0
                monthly_cost_demand = 0.0
                monthly_zones_ok = 0
                monthly_zone_checks = 0
                monthly_deviation = 0.0
                monthly_dev_steps = 0
                cs_m.reset()

            obs_raw = next_obs_raw
            current_seqs = next_seqs

        energy_kwh = ep_energy * 0.25 / 1000
        zcr = (total_zones_ok / total_zone_checks * 100) if total_zone_checks > 0 else 100.0
        mean_dev = total_deviation / total_dev_steps if total_dev_steps > 0 else 0.0
        db_rate = (db_violations / db_total * 100) if db_total > 0 else 0.0
        ep_cs     = cs_ep.mean
        ep_cs_occ = cs_ep.mean_occ

        all_rewards.append(ep_reward)
        all_energy.append(energy_kwh)
        all_zcr.append(zcr)
        all_dev.append(mean_dev)
        all_cs.append(ep_cs)
        all_cs_occ.append(ep_cs_occ)

        print("-" * 80)
        print(f"  Episode {ep+1} Summary ({phase}):")
        print(f"    Custom Reward:        {ep_reward:.2f} (mean/step: {ep_reward/steps:.4f})")
        print(f"    Original Reward:      {ep_orig_reward:.2f}")
        print(f"    Hourly Reward (sched):{ep_hlr:.2f}   [Sinergym HourlyLinearReward]")
        print(f"    Energy:               {energy_kwh:,.0f} kWh")
        _print_demand_coincidence(power_list, floor_power_rows, _floor_idx, indent="    ")
        ep_cost_energy = info.get('cost_energy_usd', 0.0)
        ep_cost_demand = info.get('cost_demand_usd', 0.0)
        ep_cost_total  = info.get('cost_total_usd', ep_cost_energy + ep_cost_demand)
        print(f"    Energy Cost:          ${ep_cost_energy:,.2f}")
        print(f"    Demand Cost:          ${ep_cost_demand:,.2f}")
        print(f"    Total Cost:           ${ep_cost_total:,.2f}")
        print(f"")
        print(f"    -- Comfort metrics --")
        print(f"    CS (active, weighted):{ep_cs:.4f}   [max = 1.0]")
        print(f"    CS (occupied only):   {ep_cs_occ:.4f}   [diagnostic]")
        print(f"    Zone-Comfort Rate:    {zcr:.1f}%  (occupied only)")
        print(f"    Mean Deviation:       {mean_dev:.3f} °C  (occupied only)")
        print(f"    Deadband Violations:  {db_rate:.1f}%")
        print(f"")
        print(f"    Buffer:               {len(agent.buffer)}")
        print(f"    Train Steps:          {agent.train_steps}")

        if len(all_rewards) >= 2:
            delta_r   = all_rewards[-1] - all_rewards[-2]
            delta_zcr = all_zcr[-1]     - all_zcr[-2]
            delta_cs  = all_cs[-1]      - all_cs[-2]
            print(f"    Reward Δ:             {delta_r:+.2f} ({'↑' if delta_r > 0 else '↓'})")
            print(f"    ZCR Δ:                {delta_zcr:+.1f}%")
            print(f"    CS Δ:                 {delta_cs:+.4f}")

        metrics_history = {
            'rewards':   all_rewards,
            'energy':    all_energy,
            'zcr':       all_zcr,
            'deviation': all_dev,
            'cs':        all_cs,
            'cs_occ':    all_cs_occ,
        }
        save_checkpoint(agent, normalizer, obs_builder, ep, metrics_history, checkpoint_dir)

        if not is_warmup:
            agent.save(os.path.join(checkpoint_dir, f"maddpg_office_ep{ep+1}.pt"))

    env.close()

    agent.save(os.path.join(checkpoint_dir, "maddpg_office_final.pt"))
    normalizer.save(os.path.join(checkpoint_dir, "maddpg_normalizer.npz"))

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    learn = slice(Config.WARMUP_EPISODES, None)
    learn_rewards = all_rewards[learn]
    if len(learn_rewards) > 0:
        print(f"  Avg Reward (learning):     {np.mean(learn_rewards):.2f}")
        print(f"  Avg Energy (learning):     {np.mean(all_energy[learn]):,.0f} kWh")
        print(f"  Avg ZCR (learning):        {np.mean(all_zcr[learn]):.1f}%")
        print(f"  Avg CS (learning):         {np.nanmean(all_cs[learn]):.4f}")
        print(f"  Avg CS_occ (learning):     {np.nanmean(all_cs_occ[learn]):.4f}")
        print(f"  Avg Deviation (learning):  {np.mean(all_dev[learn]):.3f} °C")

    return agent, normalizer


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate(agent, normalizer):
    from register_env import make_custom_env

    print("\n" + "=" * 80)
    print("MADDPG EVALUATION")
    print("=" * 80)

    base_env = make_custom_env(weather=Config.WEATHER)
    env = CustomRewardWrapper(base_env)

    obs_raw, info = env.reset()
    obs_builder = ObservationBuilder()
    obs_builder.reset()
    agent_obs = obs_builder.build_agent_obs(obs_raw)

    seq_builder = AgentSequenceBuilder(N_OCCUPIED, Config.SEQ_LEN, Config.AGENT_OBS_DIM)
    seq_builder.reset(agent_obs)
    current_seqs = seq_builder.get_current()

    ep_reward = 0.0
    ep_orig_reward = 0.0
    ep_energy = 0.0
    power_list = []
    floor_power_rows = []
    _floor_idx = _resolve_floor_idx()
    if _floor_idx is not None and max(_floor_idx) >= len(obs_raw):
        print(f"[coincidence] WARNING: floor_power_idx {_floor_idx} exceeds obs "
              f"size {len(obs_raw)}; disabling coincidence metric.")
        _floor_idx = None
    monthly_cost_energy = 0.0; monthly_cost_demand = 0.0
    prev_ce = 0.0; prev_cd = 0.0
    steps = 0
    terminated = truncated = False
    current_month = 0

    total_zones_ok = 0
    total_zone_checks = 0
    total_deviation = 0.0
    total_dev_steps = 0
    monthly_reward = 0.0
    monthly_zones_ok = 0
    monthly_zone_checks = 0
    monthly_deviation = 0.0
    monthly_dev_steps = 0
    db_violations = 0
    db_total = 0

    # CS accumulators
    cs_ep = CSAccumulator()
    cs_m  = CSAccumulator()

    # HourlyLinearReward (schedule reward) accumulators — reporting only
    ep_hlr = 0.0
    monthly_hlr = 0.0

    while not (terminated or truncated):
        action = agent.select_action(current_seqs, normalizer, add_noise=False)
        next_obs_raw, reward, terminated, truncated, info = env.step(action)

        next_agent_obs = obs_builder.build_agent_obs(next_obs_raw)
        next_seqs = seq_builder.append(next_agent_obs)

        for i in range(N_OCCUPIED):
            db_total += 1
            if action[i] >= action[N_OCCUPIED + i] - 2.0:
                db_violations += 1

        ep_reward += reward
        ep_orig_reward += info.get('original_reward', 0)
        ep_energy += info.get('power_W', 0)
        power_list.append(info.get('power_W', 0))
        if _floor_idx is not None:
            floor_power_rows.append([float(next_obs_raw[i]) * Config.FLOOR_POWER_SCALE
                                     for i in _floor_idx])
        monthly_reward += reward
        ce = info.get('cost_energy_usd', 0.0); cd = info.get('cost_demand_usd', 0.0)
        monthly_cost_energy += ce - prev_ce; prev_ce = ce
        monthly_cost_demand += cd - prev_cd; prev_cd = cd

        # CS update
        cs_ep.update(next_obs_raw)
        cs_m.update(next_obs_raw)

        # HourlyLinearReward (schedule reward) update — reporting only
        hlr, _ = compute_hourly_linear_reward(next_obs_raw)
        ep_hlr += hlr
        monthly_hlr += hlr

        zok, ztotal = compute_zone_comfort_rate(next_obs_raw, occupied_only=True)
        total_zones_ok += zok
        total_zone_checks += ztotal
        monthly_zones_ok += zok
        monthly_zone_checks += ztotal

        dev_val, dev_count = compute_mean_deviation(next_obs_raw, occupied_only=True)
        total_deviation += dev_val
        total_dev_steps += dev_count
        monthly_deviation += dev_val
        monthly_dev_steps += dev_count

        steps += 1

        m_now = int(next_obs_raw[IDX_MONTH])
        if m_now != current_month:
            if current_month > 0:
                avg_temp = float(np.mean(next_obs_raw[IDX_ZONE_TEMPS]))
                zt = next_obs_raw[IDX_ZONE_TEMPS]
                zok_now = count_zones_ok(next_obs_raw)
                season = "S" if get_seasonal_comfort(current_month, 15) == COMFORT_SUMMER else "W"
                m_zcr = (monthly_zones_ok / monthly_zone_checks * 100) if monthly_zone_checks > 0 else 100.0
                m_dev = monthly_deviation / monthly_dev_steps if monthly_dev_steps > 0 else 0.0
                m_cs  = cs_m.mean
                m_cso = cs_m.mean_occ
                m_cost = monthly_cost_energy + monthly_cost_demand
                print(f"  Month {current_month:2d} [{season}] | "
                      f"R: {monthly_reward:8.2f} | "
                      f"HLR: {monthly_hlr:9.2f} | "
                      f"Cost: ${m_cost:7,.0f} | "
                      f"T: {avg_temp:5.1f}°C [{np.min(zt):.1f}-{np.max(zt):.1f}] | "
                      f"CS: {m_cs:.3f} (occ:{m_cso:.3f}) | "
                      f"ZCR: {m_zcr:5.1f}% | Dev: {m_dev:.3f}°C | "
                      f"ZonesOK: {zok_now}/15")
            current_month = m_now
            monthly_reward = 0.0
            monthly_cost_energy = 0.0
            monthly_cost_demand = 0.0
            monthly_zones_ok = 0
            monthly_zone_checks = 0
            monthly_deviation = 0.0
            monthly_dev_steps = 0
            monthly_hlr = 0.0
            cs_m.reset()

        current_seqs = next_seqs

    env.close()

    energy_kwh = ep_energy * 0.25 / 1000
    zcr = (total_zones_ok / total_zone_checks * 100) if total_zone_checks > 0 else 100.0
    mean_dev = total_deviation / total_dev_steps if total_dev_steps > 0 else 0.0
    db_rate = (db_violations / db_total * 100) if db_total > 0 else 0.0

    ep_cs     = cs_ep.mean
    ep_cs_occ = cs_ep.mean_occ

    print("-" * 80)
    print(f"EVALUATION RESULTS:")
    print(f"  Custom Reward:         {ep_reward:.2f}")
    print(f"  Original Reward:       {ep_orig_reward:.2f}")
    print(f"  Hourly Reward (sched): {ep_hlr:.2f}   [Sinergym HourlyLinearReward]")
    print(f"  Energy:                {energy_kwh:,.0f} kWh")
    _print_demand_coincidence(power_list, floor_power_rows, _floor_idx, indent="  ")
    ev_cost_energy = info.get('cost_energy_usd', 0.0)
    ev_cost_demand = info.get('cost_demand_usd', 0.0)
    ev_cost_total  = info.get('cost_total_usd', ev_cost_energy + ev_cost_demand)
    print(f"  Energy Cost:           ${ev_cost_energy:,.2f}")
    print(f"  Demand Cost:           ${ev_cost_demand:,.2f}")
    print(f"  Total Cost:            ${ev_cost_total:,.2f}")
    print(f"")
    print(f"  -- Comfort metrics --")
    print(f"  CS (active, weighted): {ep_cs:.4f}   [max = 1.0]")
    print(f"  CS (occupied only):    {ep_cs_occ:.4f}   [diagnostic]")
    print(f"  Zone-Comfort Rate:     {zcr:.1f}%  (occupied hours only)")
    print(f"  Mean Deviation:        {mean_dev:.3f} °C  (occupied hours only)")
    print(f"  Deadband Violations:   {db_rate:.1f}%")
    print("=" * 80)

    return {
        'custom_reward':           ep_reward,
        'original_reward':         ep_orig_reward,
        'hourly_reward':           ep_hlr,
        'energy_kwh':              energy_kwh,
        'energy_cost_usd':         ev_cost_energy,
        'demand_cost_usd':         ev_cost_demand,
        'total_cost_usd':          ev_cost_total,
        'comfort_score':           float(ep_cs),
        'comfort_score_occ':       float(ep_cs_occ),
        'zone_comfort_rate':       zcr,
        'mean_deviation':          mean_dev,
        'deadband_violation_rate': db_rate,
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MADDPG Sinergym Training")
    parser.add_argument('--mode', choices=['train', 'evaluate', 'resume'],
                        default='resume',
                        help="'train' = fresh start, 'resume' = continue from checkpoint, "
                             "'evaluate' = eval only")
    parser.add_argument('--episodes', type=int, default=None,
                        help="Total episodes (overrides Config.TRAIN_EPISODES)")
    parser.add_argument('--warmup', type=int, default=None,
                        help="Number of warmup episodes using RBC (overrides "
                             "Config.WARMUP_EPISODES). Use 0 to disable warmup "
                             "entirely.")
    parser.add_argument('--no-warmup', action='store_true',
                        help="Shortcut for --warmup 0. Skips RBC warmup; "
                             "MADDPG explores from step 0.")
    parser.add_argument('--checkpoint-dir', type=str, default=Config.CHECKPOINT_DIR,
                        help="Checkpoint directory")
    parser.add_argument('--floor-power-idx', type=str, default="92,93,94", dest='floor_power_idx',
                        help="Comma-separated obs indices of per-floor HVAC power "
                             "[bot,mid,top] (e.g. '92,93,94'). Enables coincidence metric.")
    parser.add_argument('--floor-power-scale', type=float, default=0.0011111, dest='floor_power_scale',
                        help="Per-floor obs multiplier; meters report J/timestep so use "
                             "0.0011111 (=1/900s) for kW. CF is scale-invariant.")
    parser.add_argument('--warmup_rbc', choices=['proactive','reactive'],
                        default=Config.WARMUP_RBC,
                        help="Which RBC seeds the warm-start (proactive pre-conditions; "
                             "reactive does not). Use --no-warmup for none.")
    parser.add_argument('--floor-features', action='store_true', dest='floor_features',
                        help="v5: add [own floor, other floors, building] metered "
                             "power (kW) to every agent's observation (32-dim)")
    args = parser.parse_args()

    Config.WARMUP_RBC = args.warmup_rbc
    enable_floor_features(args.floor_features)
    Config.FLOOR_POWER_IDX = args.floor_power_idx
    Config.FLOOR_POWER_SCALE = args.floor_power_scale

    # Resolve warmup arg: --no-warmup wins over --warmup if both given
    warmup_override = 0 if args.no_warmup else args.warmup

    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    print(f"Device: {Config.DEVICE}")
    print(f"Architecture: MADDPG — {N_OCCUPIED} zone agents + centralized critic")
    print(f"Mode: {args.mode}")

    if args.mode == 'evaluate':
        from register_env import make_custom_env
        base_env = make_custom_env(weather=Config.WEATHER)
        env = CustomRewardWrapper(base_env)

        agent = MADDPGAgent(env)
        normalizer = AgentObsNormalizer(Config.AGENT_OBS_DIM)
        obs_builder = ObservationBuilder()

        _, _, loaded = load_checkpoint(agent, normalizer, obs_builder, args.checkpoint_dir)
        if not loaded:
            model_path = "maddpg_office_ep19.pt"
            norm_path = "maddpg_normalizer.npz"
            if os.path.exists(model_path) and os.path.exists(norm_path):
                print(f"Loading legacy files: {model_path}, {norm_path}")
                agent.load(model_path)
                normalizer.load(norm_path)
                if not normalizer.frozen:
                    normalizer.freeze()
            else:
                print("Error: No checkpoint or legacy files found.")
                exit(1)

        env.close()
        results = evaluate(agent, normalizer)

    else:
        do_resume = (args.mode == 'resume')
        agent, normalizer = train(
            resume=do_resume,
            checkpoint_dir=args.checkpoint_dir,
            total_episodes=args.episodes,
            warmup_episodes=warmup_override,
        )
        results = evaluate(agent, normalizer)

    print("\n" + "=" * 80)
    print("FINAL METRICS (MADDPG)")
    print("=" * 80)
    print(f"  Energy:                {results['energy_kwh']:,.0f} kWh")
    print(f"  Hourly Reward (sched): {results['hourly_reward']:.2f}")
    print(f"  CS (active, weighted): {results['comfort_score']:.4f}")
    print(f"  CS (occupied only):    {results['comfort_score_occ']:.4f}")
    print(f"  Zone-Comfort Rate:     {results['zone_comfort_rate']:.1f}%  (occupied only)")
    print(f"  Mean Deviation:        {results['mean_deviation']:.3f} °C  (occupied only)")
    print(f"  Deadband Violations:   {results['deadband_violation_rate']:.1f}%")
    print("=" * 80)