"""
metrics_utils.py — standalone comfort / reporting metrics for the Sinergym
OfficeMedium bridge.

Self-contained (numpy + datetime only). Lifted verbatim from the ensemble
controller's reporting helpers so the bridge can compute the SAME monthly /
comfort-score / ZCR / deviation numbers WITHOUT importing maddpg_v3 (which pulls
in ensemble_controller) or any training code — exactly the way
hourly_reward_metric.py is a small separate file.

Provides:
    COMFORT_WINTER, COMFORT_SUMMER
    get_seasonal_comfort(month, day)         -> (low, high)
    get_occupancy_state(month, day, hour)    -> 'occupied'|'pre_occupy'|...
    tou_multiplier(month, day, hour)         -> float
    compute_comfort_score(obs)               -> (score|None, info)
    compute_mean_deviation(obs, occupied_only=True)     -> (mean_dev_C, n)
    compute_zone_comfort_rate(obs, occupied_only=True)  -> (n_ok, n_total)
    CSAccumulator                            -> .update(obs) / .mean / .mean_occ / .reset()

NOTE: this does NOT provide CustomRewardWrapper (the training-shaped reward).
That lives in maddpg_v2 and is the only thing the bridge still optionally needs
maddpg for; the bridge imports it separately and degrades gracefully if absent.
"""

from datetime import datetime, date
from typing import Optional, Dict, Tuple

import numpy as np

# ── Constants (OfficeMedium, 92-dim obs) ────────────────────────────────────
N_OCCUPIED = 15

COMFORT_WINTER = (20.0, 23.5)
COMFORT_SUMMER = (23.0, 26.0)

IDX_MONTH = 0
IDX_DAY   = 1
IDX_HOUR  = 2
IDX_HVAC_DEMAND = 90
IDX_ZONE_TEMPS  = slice(9, 24)   # 15 occupied zone air temperatures

OCCUPIED_ZONES = [
    "Core_bottom", "Core_mid", "Core_top",
    "Perimeter_bot_ZN_1", "Perimeter_bot_ZN_2",
    "Perimeter_bot_ZN_3", "Perimeter_bot_ZN_4",
    "Perimeter_mid_ZN_1", "Perimeter_mid_ZN_2",
    "Perimeter_mid_ZN_3", "Perimeter_mid_ZN_4",
    "Perimeter_top_ZN_1", "Perimeter_top_ZN_2",
    "Perimeter_top_ZN_3", "Perimeter_top_ZN_4",
]


# ── Helpers ─────────────────────────────────────────────────────────────────
def get_seasonal_comfort(month: int, day: int) -> Tuple[float, float]:
    is_summer = ((month > 6) or (month == 6 and day >= 1)) and \
                ((month < 10) or (month == 9 and day <= 30))
    return COMFORT_SUMMER if is_summer else COMFORT_WINTER


def get_occupancy_state(month: int, day: int, hour: int) -> str:
    try:
        is_weekend = datetime(2024, int(month), int(day)).weekday() >= 5
    except Exception:
        is_weekend = False
    if is_weekend:       return 'unoccupied'
    if 7 <= hour <= 19:  return 'occupied'
    if 5 <= hour < 7:    return 'pre_occupy'
    if 19 < hour <= 21:  return 'post_occupy'
    return 'unoccupied'


def tou_multiplier(month: int, day: int, hour: int) -> float:
    try:
        is_weekend = date(2024, int(month), int(day)).weekday() >= 5
    except Exception:
        is_weekend = False
    if is_weekend:
        return 1.0
    h = int(hour)
    if 16 <= h < 21:                     return 3.0   # on-peak
    if (9 <= h < 16) or (21 <= h < 23):  return 1.5   # mid-peak
    return 1.0                                         # off-peak


def compute_comfort_score(obs_raw: np.ndarray) -> Tuple[Optional[float], Dict]:
    """Magnitude-aware, occupancy-weighted comfort score in [0, 1]. Returns
    (None, info) during fully unoccupied hours so they are excluded from the
    average."""
    obs_raw = np.asarray(obs_raw, dtype=np.float64)
    month = int(obs_raw[IDX_MONTH]); day = int(obs_raw[IDX_DAY]); hour = int(obs_raw[IDX_HOUR])
    occ_state = get_occupancy_state(month, day, hour)

    state_weight = {'occupied': 1.0, 'pre_occupy': 0.6,
                    'post_occupy': 0.2, 'unoccupied': 0.0}.get(occ_state, 0.0)
    if state_weight == 0.0:
        return None, {'skipped': True, 'occ_state': occ_state}

    cl, ch = get_seasonal_comfort(month, day)
    zone_temps = np.array([float(obs_raw[9 + i]) for i in range(N_OCCUPIED)])

    zone_occ = np.array([float(obs_raw[45 + i]) for i in range(N_OCCUPIED)])
    has_sensor_occ = np.any(zone_occ > 0)
    if not has_sensor_occ and occ_state == 'occupied':
        zone_occ = np.ones(N_OCCUPIED)

    scores = np.zeros(N_OCCUPIED, dtype=np.float32)
    for i in range(N_OCCUPIED):
        if zone_occ[i] <= 0 and has_sensor_occ:
            scores[i] = 1.0
            continue
        T = float(zone_temps[i])
        if cl <= T <= ch:
            scores[i] = 1.0
        else:
            viol = max(cl - T, 0.0) + max(T - ch, 0.0)
            norm_viol = min(viol / 1.0, 1.0)
            scores[i] = float(1.0 - (norm_viol ** 2))

    comfort_score = float(np.mean(scores))
    worst_idx = int(np.argmin(scores))
    return comfort_score, {
        'occ_state': occ_state,
        'state_weight': state_weight,
        'zone_scores': scores,
        'worst_zone': worst_idx,
        'worst_zone_name': OCCUPIED_ZONES[worst_idx],
        'worst_temp': float(zone_temps[worst_idx]),
        'n_violated': int(np.sum(scores < 0.70)),
        'n_in_band': int(np.sum((zone_temps >= cl) & (zone_temps <= ch))),
    }


def compute_mean_deviation(obs_raw: np.ndarray, occupied_only: bool = True) -> Tuple[float, int]:
    """Mean °C deviation from the comfort band during occupied hours."""
    obs_raw = np.asarray(obs_raw, dtype=np.float64)
    month = int(obs_raw[IDX_MONTH]); day = int(obs_raw[IDX_DAY]); hour = int(obs_raw[IDX_HOUR])
    if occupied_only and get_occupancy_state(month, day, hour) != 'occupied':
        return 0.0, 0
    cl, ch = get_seasonal_comfort(month, day)
    zt = np.array([float(obs_raw[9 + i]) for i in range(N_OCCUPIED)])
    devs = np.maximum(cl - zt, 0) + np.maximum(zt - ch, 0)
    return float(np.mean(devs)), 1


def compute_zone_comfort_rate(obs_raw: np.ndarray, occupied_only: bool = True) -> Tuple[int, int]:
    """Binary zone-comfort rate (n_in_band, n_total) during occupied hours."""
    obs_raw = np.asarray(obs_raw, dtype=np.float64)
    month = int(obs_raw[IDX_MONTH]); day = int(obs_raw[IDX_DAY]); hour = int(obs_raw[IDX_HOUR])
    if occupied_only and get_occupancy_state(month, day, hour) != 'occupied':
        return 0, 0
    cl, ch = get_seasonal_comfort(month, day)
    zt = np.array([float(obs_raw[9 + i]) for i in range(N_OCCUPIED)])
    return int(np.sum((zt >= cl) & (zt <= ch))), N_OCCUPIED


class CSAccumulator:
    """Running comfort score over a window (episode or month).

    STATE-WEIGHTED to match maddpg_v3.CSAccumulator: each active step is weighted
    by its occupancy state_weight (occupied=1.0, pre_occupy=0.6, post_occupy=0.2;
    fully-unoccupied steps are skipped). Built on compute_comfort_score.

    .mean      — Σ(cs · w) / Σ(w)   (state-weighted average; max = 1.0)
    .mean_occ  — Σ(cs) / n over fully-occupied steps only (w >= 1.0)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.cs_sum  = 0.0   # sum( cs * w )
        self.w_sum   = 0.0   # sum( w )
        self.occ_sum = 0.0   # sum( cs ) over fully-occupied steps
        self.occ_n   = 0     # count of fully-occupied steps

    def update(self, obs):
        cs, cs_info = compute_comfort_score(obs)
        if cs is None:
            return
        w = cs_info.get('state_weight', 1.0)
        self.cs_sum += cs * w
        self.w_sum  += w
        if w >= 1.0:
            self.occ_sum += cs
            self.occ_n   += 1

    @property
    def mean(self) -> float:
        return (self.cs_sum / self.w_sum) if self.w_sum > 0 else float('nan')

    @property
    def mean_occ(self) -> float:
        return (self.occ_sum / self.occ_n) if self.occ_n > 0 else float('nan')


# ============================================================================
# CustomRewardWrapper — training-shaped reward (verbatim from maddpg_v2)
# ============================================================================
# Inlined so the bridge no longer needs maddpg_v2 for the "Custom Reward". The
# weights below are maddpg_v2.Config's reward weights; the math is unchanged.

from datetime import datetime as _dt

try:
    import gymnasium as _gym
    _GYM_OK = True
except Exception:                       # gymnasium not installed -> metrics still usable
    _gym = None
    _GYM_OK = False


class RewardConfig:
    """Reward weights (from maddpg_v2.Config). Override by passing your own
    object with the same attributes to CustomRewardWrapper(env, config=...).

    CHANGE vs the original port: the separate energy (W_ENERGY) and TOU
    (W_TOU) penalties are merged into a single TOU-priced electricity term
    (W_ELEC). One knob now controls "use less, especially when it's
    expensive"; the time-of-use multiplier shapes the cost across the day.
    Peak demand (W_PEAK) is intentionally kept separate, because a demand
    charge is billed on the single highest kW of the month, not on cumulative
    energy, so it needs its own sparse signal.
    """
    W_ELEC        = 0.14   # merged energy+TOU. off-peak 0.14, on-peak 0.42 (x3)
    W_COMFORT     = 0.40
    W_ACTION      = 0.00
    W_DEADBAND    = 0.05
    W_TEMP_TARGET = 0.10
    W_PEAK        = 0.17
    # --- back-compat: kept so external code reading these doesn't break ----
    W_ENERGY      = 0.20   # DEPRECATED (folded into W_ELEC), unused in reward
    W_TOU         = 0.08   # DEPRECATED (folded into W_ELEC), unused in reward


# ---------------------------------------------------------------------------- #
#                       REAL-WORLD ELECTRICITY COST ($)                        #
# ---------------------------------------------------------------------------- #
# PG&E B-19-style commercial time-of-use rates (California). Anchored to EIA /
# market data, mid-2026: CA commercial ~26-27 c/kWh average; TOU swings from
# ~18-22 c/kWh off-peak to ~45-55 c/kWh on-peak (4-9pm). Demand charges for
# B-19 secondary run ~$20-35 per kW of monthly peak. These feed the $ cost
# report ONLY -- they do not change the (unitless) reward shaping above.
TOU_RATES_USD = {          # $ per kWh, by TOU period
    'off_peak': 0.22,
    'mid_peak': 0.32,
    'on_peak':  0.45,
}
DEMAND_CHARGE_USD_PER_KW = 26.0    # $ per kW of monthly peak demand
DEFAULT_TIMESTEP_HOURS   = 0.25    # 15-min step (EnergyPlus Timestep 4/hr)


def tou_period(month: int, day: int, hour: int) -> str:
    """Map a timestamp to its TOU period name (matches _tou_multiplier:
    on-peak 16-21 weekday, mid-peak 9-16 & 21-23 weekday, else off-peak)."""
    try:
        is_weekend = _dt(2024, int(month), int(day)).weekday() >= 5
    except Exception:
        is_weekend = False
    if is_weekend:
        return 'off_peak'
    h = int(hour)
    if 16 <= h < 21:
        return 'on_peak'
    if (9 <= h < 16) or (21 <= h < 23):
        return 'mid_peak'
    return 'off_peak'


def tou_rate_usd(month: int, day: int, hour: int) -> float:
    """$ per kWh for this timestamp under the B-19-style schedule above."""
    return TOU_RATES_USD[tou_period(month, day, hour)]


def _tou_multiplier(month: int, day: int, hour: int) -> float:
    """Typical California commercial TOU (PG&E B-19 style). Off-peak=1.0,
    mid-peak=1.5, on-peak=3.0."""
    try:
        is_weekend = _dt(2024, int(month), int(day)).weekday() >= 5
    except Exception:
        is_weekend = False
    if is_weekend:
        return 1.0
    h = int(hour)
    if 16 <= h < 21:
        return 3.0
    if (9 <= h < 16) or (21 <= h < 23):
        return 1.5
    return 1.0


if _GYM_OK:
    class CustomRewardWrapper(_gym.Wrapper):
        """
        Reward = -(
            W_ELEC      * elec_penalty     [log-norm HVAC power x TOU price]
          + W_COMFORT   * comfort_penalty  [seasonal, occupancy-gated]
          + W_ACTION    * action_penalty   [setpoint change smoothness]
          + W_DEADBAND  * deadband_penalty [htg >= clg - 2C]
          + W_PEAK      * peak_penalty     [new monthly peak increment]
        ) + W_TEMP_TARGET * midpoint_bonus [zone near comfort midpoint]

        Energy and TOU are merged into one TOU-priced electricity term
        (W_ELEC); peak demand stays separate. Also reports real-world $ cost
        in info: cost_energy_usd (TOU-priced kWh), cost_demand_usd (monthly
        peak demand charge) and cost_total_usd, all cumulative per episode.

        Sets info['custom_reward'] and info['original_reward'] (plus components).
        Adapted from maddpg_v2.CustomRewardWrapper (energy+TOU merged, cost added).
        """

        def __init__(self, env, config=None, timestep_hours=DEFAULT_TIMESTEP_HOURS):
            super().__init__(env)
            self.config = config if config is not None else RewardConfig
            self.prev_action   = None
            self.action_range  = self.action_space.high - self.action_space.low
            self.max_power     = 300_000.0
            self.running_peak_W  = 0.0
            self._peak_month_key = None
            # --- real-world $ cost accounting --------------------------------
            self.timestep_hours   = float(timestep_hours)   # kWh = kW * hours
            self.cost_energy_usd  = 0.0   # cumulative TOU energy charge
            self.cost_demand_usd  = 0.0   # cumulative monthly-peak demand charge
            self.energy_kWh_total = 0.0   # cumulative HVAC energy

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            self.prev_action   = None
            self.running_peak_W  = 0.0
            self._peak_month_key = None
            self.cost_energy_usd  = 0.0
            self.cost_demand_usd  = 0.0
            self.energy_kWh_total = 0.0
            return obs, info

        def _maybe_reset_peak(self, month: int):
            key = (2024, int(month))
            if self._peak_month_key != key:
                self._peak_month_key = key
                self.running_peak_W  = 0.0

        def step(self, action):
            obs, original_reward, terminated, truncated, info = self.env.step(action)

            month = int(obs[IDX_MONTH]); day = int(obs[IDX_DAY]); hour = int(obs[IDX_HOUR])
            cl, ch     = get_seasonal_comfort(month, day)
            zone_temps = obs[IDX_ZONE_TEMPS]
            occ_state  = get_occupancy_state(month, day, hour)
            power      = float(obs[IDX_HVAC_DEMAND])

            # 1. Electricity penalty (log-normalised power, TOU-priced)
            #    Merged energy+TOU: one term, scaled by the live TOU multiplier
            #    so a unit of power "hurts" more at mid/on-peak. occ_mult keeps
            #    the original small unoccupied bump.
            tou_mult   = _tou_multiplier(month, day, hour)        # 1.0 / 1.5 / 3.0
            norm_power = np.log1p(power / self.max_power) / np.log1p(1.0)
            norm_power = min(norm_power, 2.0)
            occ_mult   = 1.1 if occ_state == 'unoccupied' else 1.0
            elec_penalty = norm_power * tou_mult * occ_mult
            elec_weight  = self.config.W_ELEC

            # 2. Comfort penalty (4-tier occupancy)
            is_summer = get_seasonal_comfort(month, day) == COMFORT_SUMMER
            if occ_state == 'occupied':
                cold_v = np.maximum(cl - zone_temps, 0.0)
                hot_v  = np.maximum(zone_temps - ch,  0.0)
                violations = (cold_v + hot_v * 1.3) if is_summer else (cold_v * 1.3 + hot_v)
                zone_comfort_penalties = np.clip(violations / 4.0, 0.0, 1.0).astype(np.float32)
                comfort_penalty = float(np.mean(zone_comfort_penalties))
                comfort_weight  = self.config.W_COMFORT
            elif occ_state == 'pre_occupy':
                violations = (np.maximum(cl - zone_temps, 0.0) + np.maximum(zone_temps - ch, 0.0))
                zone_comfort_penalties = np.clip(violations / 5.0, 0.0, 1.0).astype(np.float32)
                comfort_penalty = float(np.mean(zone_comfort_penalties))
                comfort_weight  = self.config.W_COMFORT * 0.4
            elif occ_state == 'post_occupy':
                violations = (np.maximum(cl - 3.0 - zone_temps, 0.0) + np.maximum(zone_temps - ch - 3.0, 0.0))
                zone_comfort_penalties = np.clip(violations / 5.0, 0.0, 1.0).astype(np.float32)
                comfort_penalty = float(np.mean(zone_comfort_penalties))
                comfort_weight  = self.config.W_COMFORT * 0.15
            else:
                zone_comfort_penalties = np.zeros(N_OCCUPIED, dtype=np.float32)
                comfort_penalty = 0.0
                comfort_weight  = 0.0

            # 3. Midpoint bonus (occupied + pre-occupy)
            if occ_state == 'occupied':
                comfort_mid = (cl + ch) / 2.0
                dist = np.abs(zone_temps - comfort_mid)
                zone_midpoint_bonus = np.clip(1.0 - dist / 2.5, 0.0, 1.0).astype(np.float32)
                midpoint_bonus  = float(np.mean(zone_midpoint_bonus))
                midpoint_weight = self.config.W_TEMP_TARGET
            elif occ_state == 'pre_occupy':
                comfort_mid = (cl + ch) / 2.0
                dist = np.abs(zone_temps - comfort_mid)
                zone_midpoint_bonus = np.clip(1.0 - dist / 4.0, 0.0, 1.0).astype(np.float32)
                midpoint_bonus  = float(np.mean(zone_midpoint_bonus))
                midpoint_weight = self.config.W_TEMP_TARGET * 0.5
            else:
                zone_midpoint_bonus = np.zeros(N_OCCUPIED, dtype=np.float32)
                midpoint_bonus  = 0.0
                midpoint_weight = 0.0

            # 4. Action-smoothness penalty
            if self.prev_action is not None and occ_state in ('occupied', 'pre_occupy'):
                delta      = np.abs(action - self.prev_action)
                safe_range = np.where(self.action_range > 0, self.action_range, 1.0)
                action_penalty = float(np.mean(1.0 - np.exp(-3.0 * delta / safe_range)))
                action_weight  = self.config.W_ACTION
            else:
                action_penalty = 0.0
                action_weight  = 0.0
            self.prev_action = action.copy()

            # 5. Deadband penalty
            htg_a = action[:N_OCCUPIED]; clg_a = action[N_OCCUPIED:]
            db_viol = np.maximum(htg_a - clg_a + 2.0, 0.0)
            zone_db_penalties = np.clip(db_viol / 3.0, 0.0, 1.0).astype(np.float32)
            deadband_penalty  = float(np.mean(zone_db_penalties))

            # 6. Peak-demand penalty (sparse) + demand $ charge
            self._maybe_reset_peak(month)
            new_peak_W = max(power - self.running_peak_W, 0.0)
            if new_peak_W > 0.0:
                self.running_peak_W = power
            peak_penalty = np.log1p(new_peak_W / self.max_power) / np.log1p(1.0)
            peak_penalty = min(peak_penalty, 2.0)

            # ---- real-world $ cost (report only; not part of the reward) -----
            # Energy charge: kWh this step priced at the live TOU $/kWh.
            energy_kWh = (power / 1000.0) * self.timestep_hours
            step_energy_cost = energy_kWh * tou_rate_usd(month, day, hour)
            self.energy_kWh_total += energy_kWh
            self.cost_energy_usd  += step_energy_cost
            # Demand charge: $/kW applied to each new monthly-peak increment,
            # so the cumulative total = Σ_month (monthly peak kW) * $/kW.
            self.cost_demand_usd  += (new_peak_W / 1000.0) * DEMAND_CHARGE_USD_PER_KW
            total_cost_usd = self.cost_energy_usd + self.cost_demand_usd

            # Combine (elec term already TOU-priced; peak kept separate)
            reward = -(
                elec_weight               * elec_penalty
                + comfort_weight          * comfort_penalty
                + action_weight           * action_penalty
                + self.config.W_DEADBAND  * deadband_penalty
                + self.config.W_PEAK      * peak_penalty
            ) + midpoint_weight * midpoint_bonus

            # Per-zone decomposition
            elec_per_zone = elec_weight * elec_penalty / N_OCCUPIED
            peak_per_zone = self.config.W_PEAK * peak_penalty / N_OCCUPIED
            zone_rewards = np.zeros(N_OCCUPIED, dtype=np.float32)
            for i in range(N_OCCUPIED):
                zone_rewards[i] = -(
                    elec_per_zone
                    + comfort_weight          * zone_comfort_penalties[i]
                    + self.config.W_DEADBAND  * zone_db_penalties[i]
                    + peak_per_zone
                ) + midpoint_weight * zone_midpoint_bonus[i]

            info['custom_reward']   = reward
            info['original_reward'] = original_reward
            info['zone_rewards']    = zone_rewards
            info['elec_penalty']    = elec_penalty
            info['energy_penalty']  = elec_penalty   # back-compat alias
            info['comfort_penalty'] = comfort_penalty
            info['peak_penalty']    = peak_penalty
            info['tou_multiplier']  = tou_mult
            info['tou_period']      = tou_period(month, day, hour)
            info['running_peak_W']  = self.running_peak_W
            info['new_peak_W']      = new_peak_W
            info['power_W']         = power
            info['occ_state']       = occ_state
            # --- $ cost report ---
            info['energy_kWh']        = energy_kWh
            info['step_energy_cost']  = step_energy_cost
            info['cost_energy_usd']   = self.cost_energy_usd
            info['cost_demand_usd']   = self.cost_demand_usd
            info['cost_total_usd']    = total_cost_usd
            info['energy_kWh_total']  = self.energy_kWh_total
            return obs, reward, terminated, truncated, info
else:
    CustomRewardWrapper = None   # gymnasium unavailable


# ============================================================================
# Hourly-schedule linear reward (verbatim from hourly_reward_metric.py)
# ============================================================================
# Reporting-only reproduction of Sinergym's HourlyLinearReward, computed from
# the raw obs. Constants match register_env.py / Sinergym LinearReward defaults.
RANGE_COMFORT_WINTER = (20.0, 23.5)
RANGE_COMFORT_SUMMER = (23.0, 26.0)
SUMMER_START = (6, 1)
SUMMER_FINAL = (9, 30)

ENERGY_WEIGHT      = 0.5
LAMBDA_ENERGY      = 1.0e-4
LAMBDA_TEMPERATURE = 1.0

# HourlyLinearReward comfort-hours window (Sinergym default (9, 19)).
COMFORT_HOURS = (9, 19)

ZONE_TEMPS_SLICE = slice(9, 24)   # 15 occupied zones (reward temperature vars)
_HLR_YEAR = 2024


def _hlr_temp_range(month: int, day: int):
    """Pick summer vs winter comfort range exactly like Sinergym."""
    try:
        dt = datetime(_HLR_YEAR, month, day)
        s0 = datetime(_HLR_YEAR, *SUMMER_START)
        s1 = datetime(_HLR_YEAR, *SUMMER_FINAL)
        is_summer = s0 <= dt <= s1
    except Exception:
        is_summer = False
    return RANGE_COMFORT_SUMMER if is_summer else RANGE_COMFORT_WINTER


def compute_hourly_linear_reward(obs):
    """
    Reproduce Sinergym HourlyLinearReward for one step from the raw obs array.
    Returns (reward, terms); sign convention matches Sinergym (reward <= 0).
    """
    obs = np.asarray(obs, dtype=np.float64)
    month = int(obs[IDX_MONTH]); day = int(obs[IDX_DAY]); hour = int(obs[IDX_HOUR])

    # Energy penalty: -sum(power over energy_variables)
    power = float(obs[IDX_HVAC_DEMAND])
    energy_penalty = -power

    # Comfort penalty: -sum_zones( max(T_low - T, 0, T - T_up) )
    low, high = _hlr_temp_range(month, day)
    zt = np.asarray(obs[ZONE_TEMPS_SLICE], dtype=np.float64)
    violations = np.maximum.reduce([low - zt, np.zeros_like(zt), zt - high])
    total_violation = float(np.sum(violations))
    comfort_penalty = -total_violation

    # Hourly weight switch: comfort term vanishes outside comfort hours.
    w = ENERGY_WEIGHT if COMFORT_HOURS[0] <= hour <= COMFORT_HOURS[1] else 1.0

    energy_term  = LAMBDA_ENERGY      * w         * energy_penalty
    comfort_term = LAMBDA_TEMPERATURE * (1.0 - w) * comfort_penalty
    reward = energy_term + comfort_term

    return reward, {
        "energy_term":                 energy_term,
        "comfort_term":                comfort_term,
        "energy_penalty":              energy_penalty,
        "comfort_penalty":             comfort_penalty,
        "total_power_demand":          power,
        "total_temperature_violation": total_violation,
        "reward_weight":               w,
    }

# ============================================================================
# DEMAND / PEAK-COINCIDENCE METRICS
# ============================================================================
# These quantify how "peaky" the load is (the driver of the demand charge) and
# how much the zones sharing an air loop peak *together*. Building-level load
# factor / PAR are computable from the facility HVAC power already logged.
# The true per-zone coincidence factor needs PER-ZONE power, which the default
# observation does not expose -- floor_coincidence_factor() is provided for use
# once per-zone (or per-air-loop) power is available from EnergyPlus.

def load_factor(power_series) -> float:
    """avg(power) / peak(power) over the series, in [0, 1].
    1.0 = perfectly flat profile (cheap demand charge); low = peaky/expensive."""
    p = np.asarray(power_series, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float('nan')
    peak = p.max()
    return float(p.mean() / peak) if peak > 0 else float('nan')


def peak_to_average_ratio(power_series) -> float:
    """peak/avg = 1 / load_factor. Higher = peakier."""
    lf = load_factor(power_series)
    return float('nan') if (lf is None or lf != lf or lf == 0) else 1.0 / lf


def floor_coincidence_factor(zone_power) -> float:
    """Peak-coincidence factor for a group of zones sharing an air loop.

        CF = max_t( sum_z power[t, z] )  /  sum_z( max_t power[t, z] )

    Input: 2-D array shaped (timesteps, n_zones) of per-zone power.
    CF near 1.0 => all zones peak at the SAME time (worst case for the shared
    loop / demand charge). Lower => naturally staggered loads. This is the
    loop-level "are my zones fighting over the air handler" number.

    Requires per-zone power, which the standard obs does not provide -- feed it
    a matrix you build from a per-zone power output once it is exposed.
    """
    a = np.asarray(zone_power, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] == 0:
        return float('nan')
    coincident_peak = a.sum(axis=1).max()        # peak of the summed load
    sum_individual_peaks = a.max(axis=0).sum()   # sum of each zone's own peak
    if sum_individual_peaks <= 0:
        return float('nan')
    return float(coincident_peak / sum_individual_peaks)


class PeakDemandTracker:
    """Streaming building-level peak/peakiness tracker (per episode or month).

    Feed it facility HVAC power (W) each step; read peak_kW / avg_kW /
    load_factor / par at any time. This is the demand-relevant peakiness
    summary you can compute from the existing observation today.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._peak = 0.0
        self._sum = 0.0
        self._n = 0

    def update(self, power_W: float):
        p = float(power_W)
        self._sum += p
        self._n += 1
        if p > self._peak:
            self._peak = p

    @property
    def peak_kW(self) -> float:
        return self._peak / 1000.0

    @property
    def avg_kW(self) -> float:
        return (self._sum / self._n / 1000.0) if self._n > 0 else float('nan')

    @property
    def load_factor(self) -> float:
        return (self._sum / self._n / self._peak) if (self._n > 0 and self._peak > 0) else float('nan')

    @property
    def par(self) -> float:
        lf = self.load_factor
        return float('nan') if (lf != lf or lf == 0) else 1.0 / lf