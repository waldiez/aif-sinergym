"""
RBC Unified Controller — Configurable Granularity + Structural Dependencies
  granularity: 'zone' | 'floor' | 'single'
  structural:  True  | False

  granularity controls how many independent setpoint decisions are made:
    - 'zone'   : 15 independent (htg, clg) pairs — one per zone
    - 'floor'  : 3 independent (htg, clg) pairs — one per floor, broadcast to zones
    - 'single' : 1 (htg, clg) pair — broadcast to all 15 zones

  structural controls whether the controller accounts for building structure:
    - True  : adjustments for core/perimeter, orientation, floor position,
              neighboring zone influence, inter-floor coupling
    - False : pure temperature-vs-comfort-band logic only
"""

from datetime import datetime
from typing import Dict
import numpy as np


# =============================================================================
# ZONE CONFIGURATION
# =============================================================================
OCCUPIED_ZONES = [
    "Core_bottom", "Core_mid", "Core_top",
    "Perimeter_bot_ZN_1", "Perimeter_bot_ZN_2",
    "Perimeter_bot_ZN_3", "Perimeter_bot_ZN_4",
    "Perimeter_mid_ZN_1", "Perimeter_mid_ZN_2",
    "Perimeter_mid_ZN_3", "Perimeter_mid_ZN_4",
    "Perimeter_top_ZN_1", "Perimeter_top_ZN_2",
    "Perimeter_top_ZN_3", "Perimeter_top_ZN_4",
]

N_OCCUPIED = len(OCCUPIED_ZONES)

ZONE_META = {
    0:  {"name": "Core_bottom",        "floor": "bottom", "type": "core",      "orient": "none"},
    1:  {"name": "Core_mid",           "floor": "mid",    "type": "core",      "orient": "none"},
    2:  {"name": "Core_top",           "floor": "top",    "type": "core",      "orient": "none"},
    3:  {"name": "Perimeter_bot_ZN_1", "floor": "bottom", "type": "perimeter", "orient": "south"},
    4:  {"name": "Perimeter_bot_ZN_2", "floor": "bottom", "type": "perimeter", "orient": "east"},
    5:  {"name": "Perimeter_bot_ZN_3", "floor": "bottom", "type": "perimeter", "orient": "north"},
    6:  {"name": "Perimeter_bot_ZN_4", "floor": "bottom", "type": "perimeter", "orient": "west"},
    7:  {"name": "Perimeter_mid_ZN_1", "floor": "mid",    "type": "perimeter", "orient": "south"},
    8:  {"name": "Perimeter_mid_ZN_2", "floor": "mid",    "type": "perimeter", "orient": "east"},
    9:  {"name": "Perimeter_mid_ZN_3", "floor": "mid",    "type": "perimeter", "orient": "north"},
    10: {"name": "Perimeter_mid_ZN_4", "floor": "mid",    "type": "perimeter", "orient": "west"},
    11: {"name": "Perimeter_top_ZN_1", "floor": "top",    "type": "perimeter", "orient": "south"},
    12: {"name": "Perimeter_top_ZN_2", "floor": "top",    "type": "perimeter", "orient": "east"},
    13: {"name": "Perimeter_top_ZN_3", "floor": "top",    "type": "perimeter", "orient": "north"},
    14: {"name": "Perimeter_top_ZN_4", "floor": "top",    "type": "perimeter", "orient": "west"},
}

FLOOR_ZONES = {
    "bottom": [i for i, m in ZONE_META.items() if m["floor"] == "bottom"],
    "mid":    [i for i, m in ZONE_META.items() if m["floor"] == "mid"],
    "top":    [i for i, m in ZONE_META.items() if m["floor"] == "top"],
}

FLOORS = ["bottom", "mid", "top"]

# ---- Structural adjacency maps ----
# Same-floor neighbors: each perimeter zone is adjacent to core + neighboring perimeters
# Core is adjacent to all perimeters on its floor
SAME_FLOOR_NEIGHBORS = {
    # bottom
    0: [3, 4, 5, 6],       # core_bottom -> all bottom perimeters
    3: [0, 4, 6],           # perim_bot_1 (south) -> core, east, west
    4: [0, 3, 5],           # perim_bot_2 (east)  -> core, south, north
    5: [0, 4, 6],           # perim_bot_3 (north) -> core, east, west
    6: [0, 3, 5],           # perim_bot_4 (west)  -> core, south, north
    # mid
    1: [7, 8, 9, 10],
    7: [1, 8, 10],
    8: [1, 7, 9],
    9: [1, 8, 10],
    10: [1, 7, 9],
    # top
    2: [11, 12, 13, 14],
    11: [2, 12, 14],
    12: [2, 11, 13],
    13: [2, 12, 14],
    14: [2, 11, 13],
}

# Vertical neighbors: maps zone -> zone directly above/below (same position, different floor)
VERTICAL_NEIGHBORS = {
    # bottom <-> mid
    0: [1], 1: [0, 2], 2: [1],
    3: [7], 7: [3, 11], 11: [7],
    4: [8], 8: [4, 12], 12: [8],
    5: [9], 9: [5, 13], 13: [9],
    6: [10], 10: [6, 14], 14: [10],
}

# Floor vertical neighbors
FLOOR_VERTICAL = {
    "bottom": ["mid"],
    "mid":    ["bottom", "top"],
    "top":    ["mid"],
}


# =============================================================================
# OBSERVATION INDICES
# =============================================================================
class ObsIndex:
    MONTH = 0
    DAY = 1
    HOUR = 2
    OUTDOOR_TEMP = 3
    DIRECT_SOLAR = 8
    ZONE_TEMPS_START = 9
    ZONE_TEMPS_END = 27
    HVAC_DEMAND = 90

    @staticmethod
    def get_zone_temp(obs, zone_idx):
        return float(obs[9 + zone_idx])

    @staticmethod
    def get_all_htg_setpoints(obs):
        return np.array([obs[60 + i * 2] for i in range(N_OCCUPIED)])

    @staticmethod
    def get_all_clg_setpoints(obs):
        return np.array([obs[60 + i * 2 + 1] for i in range(N_OCCUPIED)])


# =============================================================================
# COMFORT RANGES
# =============================================================================
COMFORT_WINTER = (20.0, 23.5)
COMFORT_SUMMER = (23.0, 26.0)


def get_seasonal_comfort(month, day):
    is_summer = (month > 6 or (month == 6 and day >= 1)) and \
                (month < 10 or (month == 9 and day <= 30))
    return COMFORT_SUMMER if is_summer else COMFORT_WINTER


# =============================================================================
# OCCUPANCY
# =============================================================================
def get_occupancy_state(month, day, hour):
    try:
        is_weekend = datetime(2024, month, day).weekday() >= 5
    except:
        is_weekend = False
    if is_weekend:
        return 'unoccupied'
    if 6 <= hour <= 20:
        return 'occupied'
    if 4 <= hour < 6:
        return 'pre_occupied'
    return 'unoccupied'


def is_occupied_hour(obs):
    month = int(obs[ObsIndex.MONTH])
    day = int(obs[ObsIndex.DAY])
    hour = int(obs[ObsIndex.HOUR])
    return get_occupancy_state(month, day, hour) == 'occupied'


# =============================================================================
# OCCUPANCY-AWARE METRICS
# =============================================================================
def compute_zone_comfort_rate(obs, occupied_only=True):
    if occupied_only and not is_occupied_hour(obs):
        return 0, 0
    month, day = int(obs[ObsIndex.MONTH]), int(obs[ObsIndex.DAY])
    cl, ch = get_seasonal_comfort(month, day)
    zt = obs[ObsIndex.ZONE_TEMPS_START:ObsIndex.ZONE_TEMPS_START + N_OCCUPIED]
    return int(np.sum((zt >= cl) & (zt <= ch))), N_OCCUPIED


def compute_mean_deviation(obs, occupied_only=True):
    if occupied_only and not is_occupied_hour(obs):
        return 0.0, 0
    month, day = int(obs[ObsIndex.MONTH]), int(obs[ObsIndex.DAY])
    cl, ch = get_seasonal_comfort(month, day)
    zt = obs[ObsIndex.ZONE_TEMPS_START:ObsIndex.ZONE_TEMPS_START + N_OCCUPIED]
    deviations = np.maximum(cl - zt, 0) + np.maximum(zt - ch, 0)
    return float(np.mean(deviations)), 1


def count_zones_ok(obs):
    month, day = int(obs[ObsIndex.MONTH]), int(obs[ObsIndex.DAY])
    cl, ch = get_seasonal_comfort(month, day)
    zt = obs[ObsIndex.ZONE_TEMPS_START:ObsIndex.ZONE_TEMPS_START + N_OCCUPIED]
    return int(np.sum((zt >= cl) & (zt <= ch)))


# =============================================================================
# METRICS HELPER
# =============================================================================
class AgentMetrics:
    def __init__(self, name):
        self.name = name
        self.rewards = []
        self.temperatures = []

    def record(self, reward, temperature=None):
        self.rewards.append(reward)
        if temperature is not None:
            self.temperatures.append(temperature)

    def reset(self):
        self.rewards = []
        self.temperatures = []


# =============================================================================
# UNIFIED CONTROLLER
# =============================================================================
class RuleBasedController:
    """
    Unified RBC with configurable granularity and structural awareness.

    Parameters
    ----------
    env : gymnasium.Env
    granularity : str
        'zone'   -> per-zone setpoints (15 decisions)
        'floor'  -> per-floor setpoints (3 decisions, broadcast)
        'single' -> single setpoint pair (1 decision, broadcast to all)
    structural : bool
        True  -> use core/perimeter, orientation, floor position,
                 neighbor influence, inter-floor coupling
        False -> pure temperature-vs-comfort logic only
    """

    VALID_GRANULARITIES = ("zone", "floor", "single")

    def __init__(self, env, granularity="zone", structural=True, cool_margin=1.0):
        assert granularity in self.VALID_GRANULARITIES, \
            f"granularity must be one of {self.VALID_GRANULARITIES}, got '{granularity}'"
        self.env = env
        self.granularity = granularity
        # FULLY-REACTIVE VARIANT: structural overrides are disabled regardless
        # of the flag, so behaviour is pure comfort-band logic only.
        self.structural = False
        # Cooling target margin above the comfort floor (same as proactive):
        # cooling is never held below (comfort_low + cool_margin), so summer
        # zones settle ~24C inside the 23-26 band instead of on the 23 edge.
        self.cool_margin = float(cool_margin)
        self.n_zones = N_OCCUPIED
        self.heat_low = float(env.action_space.low[0])
        self.heat_high = float(env.action_space.high[0])
        self.cool_low = float(env.action_space.low[self.n_zones])
        self.cool_high = float(env.action_space.high[self.n_zones])
        label = f"RBC-{granularity}-reactive"
        self.metrics = AgentMetrics(name=label)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def get_average_zone_temperature(self, obs):
        zt = obs[ObsIndex.ZONE_TEMPS_START:ObsIndex.ZONE_TEMPS_START + self.n_zones]
        return float(np.mean(zt))

    def get_zone_temperatures(self, obs):
        return obs[ObsIndex.ZONE_TEMPS_START:ObsIndex.ZONE_TEMPS_START + self.n_zones]

    def _get_floor_avg_temp(self, obs, floor_name):
        temps = [ObsIndex.get_zone_temp(obs, i) for i in FLOOR_ZONES[floor_name]]
        return float(np.mean(temps))

    def _get_neighbor_avg_temp(self, obs, zone_idx):
        """Average temp of same-floor + vertical neighbors."""
        neighbors = SAME_FLOOR_NEIGHBORS.get(zone_idx, []) + \
                    VERTICAL_NEIGHBORS.get(zone_idx, [])
        if not neighbors:
            return ObsIndex.get_zone_temp(obs, zone_idx)
        return float(np.mean([ObsIndex.get_zone_temp(obs, n) for n in neighbors]))

    def _get_floor_neighbor_avg_temp(self, obs, floor_name):
        """Average temp of vertically adjacent floors."""
        adj_floors = FLOOR_VERTICAL.get(floor_name, [])
        if not adj_floors:
            return self._get_floor_avg_temp(obs, floor_name)
        temps = []
        for f in adj_floors:
            temps.extend([ObsIndex.get_zone_temp(obs, i) for i in FLOOR_ZONES[f]])
        return float(np.mean(temps))

    def _clamp(self, htg, clg, comfort_low=None, comfort_high=None):
        # Season-aware bounds: cooling never below (comfort_low + cool_margin)
        # so summer zones settle inside the 23-26 band (around 24) instead of
        # on the 23 edge; heating never above comfort_high so winter zones
        # don't overheat above 23.5. Keeps every setpoint inside the band.
        if comfort_low is None:
            clg_floor = self.cool_low
        else:
            target = comfort_low + self.cool_margin
            target = min(target, comfort_high - 0.5) if comfort_high is not None else target
            clg_floor = float(np.clip(target, self.cool_low, self.cool_high))
        htg_cap = self.heat_high if comfort_high is None \
            else float(np.clip(comfort_high, self.heat_low, self.heat_high))
        return (np.clip(htg, self.heat_low, htg_cap),
                np.clip(clg, clg_floor, self.cool_high))

    # ------------------------------------------------------------------
    # base setpoint logic (no structural awareness)
    # ------------------------------------------------------------------
    def _base_setpoints_occupied(self, temp, comfort_low, comfort_high):
        if temp < comfort_low:
            htg, clg = self.heat_high, self.cool_high
        elif temp > comfort_high:
            htg, clg = self.heat_low, self.cool_low
        elif temp > comfort_high - 0.3:
            htg, clg = self.heat_low, self.cool_low
        elif temp < comfort_low + 0.3:
            htg, clg = self.heat_high, self.cool_high
        elif temp > (comfort_low + comfort_high) / 2.0:
            htg = self.heat_low
            clg = np.clip(comfort_high - 0.5, self.cool_low, self.cool_high)
        else:
            htg = np.clip(comfort_low + 0.5, self.heat_low, self.heat_high)
            clg = self.cool_high
        return htg, clg

    def _base_setpoints_precondition(self, temp, comfort_low, comfort_high,
                                      outdoor_temp):
        if temp < comfort_low:
            htg, clg = self.heat_high, self.cool_high
        elif temp > comfort_high:
            htg, clg = self.heat_low, self.cool_low
        else:
            htg = np.clip(comfort_low, self.heat_low, self.heat_high)
            clg = np.clip(comfort_high, self.cool_low, self.cool_high)
        if outdoor_temp < 10.0:
            htg = self.heat_high
        return htg, clg

    def _base_setpoints_unoccupied(self, temp, comfort_low, comfort_high):
        setback = 2.0
        sb_low = comfort_low - setback
        sb_high = comfort_high + setback
        if temp < sb_low:
            htg, clg = self.heat_high, self.cool_high
        elif temp > sb_high:
            htg, clg = self.heat_low, self.cool_low
        else:
            htg = np.clip(sb_low, self.heat_low, self.heat_high)
            clg = np.clip(sb_high, self.cool_low, self.cool_high)
        return htg, clg

    # ------------------------------------------------------------------
    # structural adjustments — zone level
    # ------------------------------------------------------------------
    def _apply_zone_structural(self, htg, clg, obs, zone_idx,
                                comfort_low, comfort_high,
                                outdoor_temp, direct_solar, hour):
        """Modify setpoints based on zone structural properties and neighbors."""
        meta = ZONE_META[zone_idx]
        zone_temp = ObsIndex.get_zone_temp(obs, zone_idx)
        is_core = meta["type"] == "core"
        orient = meta["orient"]
        floor = meta["floor"]

        # --- Core zones: internal gains dominate ---
        if is_core:
            if zone_temp > comfort_low + 0.5:
                htg = self.heat_low
            clg = self.cool_low   # always keep cooling ready
            return htg, clg

        # --- Perimeter orientation ---
        if orient == "north" and outdoor_temp < 15.0:
            htg = self.heat_high
        elif orient == "south":
            if direct_solar > 200 and zone_temp > comfort_low + 0.5:
                htg = self.heat_low
            if direct_solar > 200 and zone_temp > comfort_high - 1.0:
                clg = self.cool_low
        elif orient == "east":
            if hour < 13 and direct_solar > 200 and zone_temp > comfort_low + 0.5:
                htg = self.heat_low
            if hour < 13 and direct_solar > 200 and zone_temp > comfort_high - 1.0:
                clg = self.cool_low
        elif orient == "west":
            if hour >= 12 and direct_solar > 200 and zone_temp > comfort_low + 0.5:
                htg = self.heat_low
            if hour >= 12 and direct_solar > 200 and zone_temp > comfort_high - 1.0:
                clg = self.cool_low

        # --- Floor position ---
        if floor == "top":
            if outdoor_temp < 5.0:
                htg = self.heat_high
            if outdoor_temp > 20.0 or zone_temp > comfort_high - 0.5:
                clg = self.cool_low
        elif floor == "bottom" and outdoor_temp < 5.0:
            htg = self.heat_high

        # --- Neighbor influence ---
        # If neighbors are significantly warmer/cooler, nudge setpoints
        neighbor_avg = self._get_neighbor_avg_temp(obs, zone_idx)
        delta = neighbor_avg - zone_temp
        # Neighbors warmer -> this zone will receive heat transfer -> reduce heating
        if delta > 1.5 and htg > self.heat_low:
            htg = max(htg - 1.0, self.heat_low)
        # Neighbors cooler -> this zone will lose heat -> increase heating
        if delta < -1.5 and htg < self.heat_high:
            htg = min(htg + 1.0, self.heat_high)
        # Neighbors warmer -> this zone may warm up -> lower cooling setpoint
        if delta > 1.5 and zone_temp > comfort_low + 1.0:
            clg = max(clg - 0.5, self.cool_low)

        return htg, clg

    # ------------------------------------------------------------------
    # structural adjustments — floor level
    # ------------------------------------------------------------------
    def _apply_floor_structural(self, htg, clg, obs, floor_name,
                                 comfort_low, comfort_high,
                                 outdoor_temp, direct_solar, hour):
        """Modify setpoints based on floor-level structural properties."""
        avg_temp = self._get_floor_avg_temp(obs, floor_name)

        # --- Floor position effects ---
        if floor_name == "top":
            if outdoor_temp < 5.0:
                htg = self.heat_high
            if outdoor_temp > 20.0 or avg_temp > comfort_high - 0.5:
                clg = self.cool_low
        elif floor_name == "bottom":
            if outdoor_temp < 5.0:
                htg = self.heat_high

        # --- Inter-floor coupling ---
        neighbor_floor_avg = self._get_floor_neighbor_avg_temp(obs, floor_name)
        delta = neighbor_floor_avg - avg_temp
        if delta > 1.5 and htg > self.heat_low:
            htg = max(htg - 1.0, self.heat_low)
        if delta < -1.5 and htg < self.heat_high:
            htg = min(htg + 1.0, self.heat_high)
        if delta > 1.5 and avg_temp > comfort_low + 1.0:
            clg = max(clg - 0.5, self.cool_low)

        # --- Mid floor benefits from insulation of floors above/below ---
        if floor_name == "mid":
            # Mid floor is more thermally stable, can be less aggressive
            midpoint = (comfort_low + comfort_high) / 2.0
            if abs(avg_temp - midpoint) < 1.0:
                # Well within range — relax slightly toward energy savings
                htg = max(htg - 0.5, self.heat_low)
                clg = min(clg + 0.5, self.cool_high)

        return htg, clg

    # ------------------------------------------------------------------
    # structural adjustments — single level (building-wide)
    # ------------------------------------------------------------------
    def _apply_single_structural(self, htg, clg, obs,
                                  comfort_low, comfort_high,
                                  outdoor_temp, direct_solar, hour):
        """Modify the single setpoint pair with building-wide structural info."""
        zt = self.get_zone_temperatures(obs)
        avg_temp = float(np.mean(zt))
        temp_spread = float(np.max(zt) - np.min(zt))

        # Large temperature spread means some zones are struggling
        # -> be more aggressive on the dominant need
        if temp_spread > 3.0:
            cold_count = np.sum(zt < comfort_low)
            hot_count = np.sum(zt > comfort_high)
            if cold_count > hot_count:
                htg = self.heat_high
            elif hot_count > cold_count:
                clg = self.cool_low

        # Top floor exposure effect on building average
        top_avg = self._get_floor_avg_temp(obs, "top")
        if outdoor_temp < 5.0 and top_avg < comfort_low:
            htg = self.heat_high
        if outdoor_temp > 25.0 and top_avg > comfort_high - 0.5:
            clg = self.cool_low

        return htg, clg

    # ------------------------------------------------------------------
    # main action
    # ------------------------------------------------------------------
    def get_action(self, obs, info=None):
        month = int(obs[ObsIndex.MONTH])
        day = int(obs[ObsIndex.DAY])
        hour = int(obs[ObsIndex.HOUR])
        outdoor_temp = float(obs[ObsIndex.OUTDOOR_TEMP])
        direct_solar = float(obs[ObsIndex.DIRECT_SOLAR])
        comfort_low, comfort_high = get_seasonal_comfort(month, day)
        occ_state = get_occupancy_state(month, day, hour)

        # FULLY-REACTIVE VARIANT: no pre-conditioning. Pre-occupancy hours are
        # treated as unoccupied (setback), so the controller never anticipates
        # upcoming occupancy — it only reacts to the current zone temperature.
        if occ_state == 'pre_occupied':
            occ_state = 'unoccupied'

        action = np.zeros(self.n_zones * 2, dtype=np.float32)

        if self.granularity == "zone":
            self._action_per_zone(action, obs, occ_state,
                                   comfort_low, comfort_high,
                                   outdoor_temp, direct_solar, hour)
        elif self.granularity == "floor":
            self._action_per_floor(action, obs, occ_state,
                                    comfort_low, comfort_high,
                                    outdoor_temp, direct_solar, hour)
        else:  # single
            self._action_single(action, obs, occ_state,
                                 comfort_low, comfort_high,
                                 outdoor_temp, direct_solar, hour)
        return action

    def _action_per_zone(self, action, obs, occ_state,
                          comfort_low, comfort_high,
                          outdoor_temp, direct_solar, hour):
        for i in range(self.n_zones):
            zone_temp = ObsIndex.get_zone_temp(obs, i)
            if occ_state == 'occupied':
                htg, clg = self._base_setpoints_occupied(
                    zone_temp, comfort_low, comfort_high)
            elif occ_state == 'pre_occupied':
                htg, clg = self._base_setpoints_precondition(
                    zone_temp, comfort_low, comfort_high, outdoor_temp)
            else:
                htg, clg = self._base_setpoints_unoccupied(
                    zone_temp, comfort_low, comfort_high)

            if self.structural:
                htg, clg = self._apply_zone_structural(
                    htg, clg, obs, i, comfort_low, comfort_high,
                    outdoor_temp, direct_solar, hour)

            htg, clg = self._clamp(htg, clg, comfort_low, comfort_high)
            action[i] = htg
            action[self.n_zones + i] = clg

    def _action_per_floor(self, action, obs, occ_state,
                           comfort_low, comfort_high,
                           outdoor_temp, direct_solar, hour):
        for floor_name in FLOORS:
            avg_temp = self._get_floor_avg_temp(obs, floor_name)
            if occ_state == 'occupied':
                htg, clg = self._base_setpoints_occupied(
                    avg_temp, comfort_low, comfort_high)
            elif occ_state == 'pre_occupied':
                htg, clg = self._base_setpoints_precondition(
                    avg_temp, comfort_low, comfort_high, outdoor_temp)
            else:
                htg, clg = self._base_setpoints_unoccupied(
                    avg_temp, comfort_low, comfort_high)

            if self.structural:
                htg, clg = self._apply_floor_structural(
                    htg, clg, obs, floor_name, comfort_low, comfort_high,
                    outdoor_temp, direct_solar, hour)

            htg, clg = self._clamp(htg, clg, comfort_low, comfort_high)
            for zone_idx in FLOOR_ZONES[floor_name]:
                action[zone_idx] = htg
                action[self.n_zones + zone_idx] = clg

    def _action_single(self, action, obs, occ_state,
                        comfort_low, comfort_high,
                        outdoor_temp, direct_solar, hour):
        avg_temp = self.get_average_zone_temperature(obs)
        if occ_state == 'occupied':
            htg, clg = self._base_setpoints_occupied(
                avg_temp, comfort_low, comfort_high)
        elif occ_state == 'pre_occupied':
            htg, clg = self._base_setpoints_precondition(
                avg_temp, comfort_low, comfort_high, outdoor_temp)
        else:
            htg, clg = self._base_setpoints_unoccupied(
                avg_temp, comfort_low, comfort_high)

        if self.structural:
            htg, clg = self._apply_single_structural(
                htg, clg, obs, comfort_low, comfort_high,
                outdoor_temp, direct_solar, hour)

        htg, clg = self._clamp(htg, clg, comfort_low, comfort_high)
        action[:self.n_zones] = htg
        action[self.n_zones:] = clg

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    def record_outcome(self, reward, next_temp):
        self.metrics.record(reward, next_temp)

    def reset(self):
        self.metrics.reset()


# =============================================================================
# OCCUPIED-ONLY CUSTOM REWARD CALCULATOR
# =============================================================================
class OccupiedRewardCalculator:
    """
    Mirrors the env reward structure but ONLY penalizes comfort when occupied.
    Energy is always penalized. Comfort, midpoint bonus, deadband, and action
    smoothness are only counted during occupied hours.

    Weights:
        W_ENERGY     = 0.25
        W_COMFORT    = 0.45
        W_ACTION     = 0.08
        W_DEADBAND   = 0.07
        W_TEMP_TARGET = 0.15
    """

    W_ENERGY     = 0.25
    W_COMFORT    = 0.45
    W_ACTION     = 0.08
    W_DEADBAND   = 0.07
    W_TEMP_TARGET = 0.15
    MAX_POWER    = 300_000.0

    def __init__(self, action_space):
        self.action_range = action_space.high - action_space.low
        self.prev_action = None

    def reset(self):
        self.prev_action = None

    def compute(self, obs, action):
        month = int(obs[ObsIndex.MONTH])
        day = int(obs[ObsIndex.DAY])
        hour = int(obs[ObsIndex.HOUR])
        cl, ch = get_seasonal_comfort(month, day)
        zone_temps = obs[ObsIndex.ZONE_TEMPS_START:ObsIndex.ZONE_TEMPS_START + N_OCCUPIED]
        occ_state = get_occupancy_state(month, day, hour)
        is_occupied = (occ_state == 'occupied')

        # --- Energy penalty (always active) ---
        power = float(obs[ObsIndex.HVAC_DEMAND])
        energy_penalty = np.log1p(power / self.MAX_POWER) / np.log1p(1.0)
        energy_penalty = min(energy_penalty, 2.0)

        # --- Comfort penalty (occupied only) ---
        if is_occupied:
            is_summer = get_seasonal_comfort(month, day) == COMFORT_SUMMER
            cold_v = np.maximum(cl - zone_temps, 0.0)
            hot_v = np.maximum(zone_temps - ch, 0.0)
            if is_summer:
                violations = cold_v + hot_v * 1.3
            else:
                violations = cold_v * 1.3 + hot_v
            norm_v = np.clip(violations / 4.0, 0.0, 1.0)
            comfort_penalty = float(np.mean(norm_v))
            comfort_weight = self.W_COMFORT
        else:
            comfort_penalty = 0.0
            comfort_weight = 0.0

        # --- Midpoint bonus (occupied only) ---
        if is_occupied:
            comfort_mid = (cl + ch) / 2.0
            dist = np.abs(zone_temps - comfort_mid)
            zone_midpoint = np.clip(1.0 - dist / 2.5, 0.0, 1.0)
            midpoint_bonus = float(np.mean(zone_midpoint))
            midpoint_weight = self.W_TEMP_TARGET
        else:
            midpoint_bonus = 0.0
            midpoint_weight = 0.0

        # --- Action smoothness (occupied only) ---
        if self.prev_action is not None and is_occupied:
            delta = np.abs(action - self.prev_action)
            safe_range = np.where(self.action_range > 0, self.action_range, 1.0)
            action_penalty = float(np.mean(1.0 - np.exp(-3.0 * delta / safe_range)))
            action_weight = self.W_ACTION
        else:
            action_penalty = 0.0
            action_weight = 0.0
        self.prev_action = action.copy()

        # --- Deadband penalty (occupied only) ---
        if is_occupied:
            htg_a = action[:N_OCCUPIED]
            clg_a = action[N_OCCUPIED:]
            db_viol = np.maximum(htg_a - clg_a + 2.0, 0.0)
            deadband_penalty = float(np.mean(np.clip(db_viol / 3.0, 0, 1)))
            deadband_weight = self.W_DEADBAND
        else:
            deadband_penalty = 0.0
            deadband_weight = 0.0

        # --- Total reward ---
        reward = -(
            self.W_ENERGY * energy_penalty +
            comfort_weight * comfort_penalty +
            action_weight * action_penalty +
            deadband_weight * deadband_penalty
        ) + midpoint_weight * midpoint_bonus

        return reward, {
            'energy_penalty': energy_penalty,
            'comfort_penalty': comfort_penalty,
            'midpoint_bonus': midpoint_bonus,
            'action_penalty': action_penalty,
            'deadband_penalty': deadband_penalty,
            'is_occupied': is_occupied,
            'power_W': power,
        }


# =============================================================================
# SIMULATION RUNNER
# =============================================================================
def run_simulation(weather="mixed", episodes=1, granularity="zone", structural=True,
                   cool_margin=1.0, floor_power_idx=None, floor_power_scale=1.0):
    from register_env import make_custom_env
    # All scoring/reward/cost come from metrics_utils (single source of truth,
    # shared with the AIF and MADDPG runs) -- NOT from custom_reward_metric /
    # hourly_reward_metric / maddpg.
    from metrics_utils import (
        CustomRewardWrapper,
        CSAccumulator,
        compute_hourly_linear_reward,
        compute_zone_comfort_rate,
        compute_mean_deviation,
        load_factor,
        floor_coincidence_factor,
        DEFAULT_TIMESTEP_HOURS,
    )

    # Per-floor coincidence factor (game-theoretic metric across the 3 air-loop
    # floors sharing the building's electric service). Pass obs indices
    # [bot,mid,top]; Sinergym meters report Joules/timestep, so use
    # floor_power_scale=0.0011111 (=1/900s) for kW. CF itself is scale-invariant.
    if isinstance(floor_power_idx, str):
        floor_power_idx = [int(x) for x in floor_power_idx.split(",") if x.strip()]
    if floor_power_idx is not None and len(floor_power_idx) != 3:
        print(f"[coincidence] WARNING: expected 3 floor-power indices, got "
              f"{floor_power_idx}; disabling coincidence metric.")
        floor_power_idx = None

    # real_world=False -> base env returns Sinergym LinearReward (metric #1),
    # which the wrapper preserves as info['original_reward']. The wrapper's
    # returned reward is the custom weighted reward (metric #2), and it also
    # reports the real-world $ cost (energy TOU + monthly demand) in info.
    env = CustomRewardWrapper(
        make_custom_env(weather=weather, real_world=False),
        timestep_hours=DEFAULT_TIMESTEP_HOURS,
    )

    struct_label = "FULLY-REACTIVE (no pre-conditioning, no structural)"
    print(f"{'='*80}")
    print(f"RBC Reactive | granularity={granularity} | {struct_label} | Weather: {weather}")
    print(f"  Heating: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    print(f"  Cooling: [{env.action_space.low[N_OCCUPIED]}, {env.action_space.high[N_OCCUPIED]}]")
    print(f"  Comfort Winter: {COMFORT_WINTER}  Summer: {COMFORT_SUMMER}")
    print(f"  Metrics: Occupied hours only (weekday 6:00-20:00)")
    print(f"  Rewards: Sinergym LinearReward (env) + Custom weighted + Sinergym HourlyLinearReward")
    print(f"{'='*80}\n")

    agent = RuleBasedController(env, granularity=granularity, structural=structural,
                                cool_margin=cool_margin)

    all_rewards = []      # Sinergym LinearReward (env reward)
    all_custom_rewards = []   # custom weighted reward
    all_hlr_rewards = []      # Sinergym HourlyLinearReward
    all_energy = []
    all_zcr = []
    all_dev = []
    all_cost = []         # total $ cost per episode
    all_cs = []           # comfort score (weighted)
    all_cs_occ = []       # comfort score (occupied-only)

    for ep in range(episodes):
        obs, info = env.reset()
        if floor_power_idx is not None and max(floor_power_idx) >= len(obs):
            print(f"[coincidence] WARNING: floor_power_idx {floor_power_idx} exceeds "
                  f"observation size {len(obs)} (valid 0-{len(obs)-1}). Per-floor power "
                  f"is not in the observation; disabling coincidence metric.")
            floor_power_idx = None
        floor_power_rows = []
        rewards = []
        custom_rewards = []
        hlr_rewards = []
        energy_list = []
        terminated = truncated = False
        current_month = 0
        steps = 0
        # --- real-world $ cost (cumulative in info; take per-month deltas) ---
        monthly_cost_energy = 0.0
        monthly_cost_demand = 0.0
        prev_ce = 0.0
        prev_cd = 0.0
        cs_ep = CSAccumulator()   # comfort score (episode)
        cs_m  = CSAccumulator()   # comfort score (monthly)

        total_zones_ok = 0
        total_zone_checks = 0
        total_deviation = 0.0
        total_dev_steps = 0

        monthly_rewards = []
        monthly_custom_rewards = []
        monthly_hlr_rewards = []
        monthly_energy = 0.0
        monthly_zones_ok = 0
        monthly_zone_checks = 0
        monthly_deviation = 0.0
        monthly_dev_steps = 0
        monthly_steps = 0

        print(f"Episode {ep+1}/{episodes}")
        print("-" * 80)

        while not (terminated or truncated):
            action = agent.get_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)

            # Wrapper returns custom reward; Sinergym LinearReward is preserved.
            lin_r    = info.get('original_reward', reward)   # metric #1
            custom_r = info.get('custom_reward',   reward)   # metric #2

            # Env reward = Sinergym LinearReward (metric #1)
            rewards.append(lin_r)
            monthly_rewards.append(lin_r)
            agent.record_outcome(lin_r, agent.get_average_zone_temperature(obs))

            # Custom weighted reward (metric #2) — from metrics_utils wrapper
            custom_rewards.append(custom_r)
            monthly_custom_rewards.append(custom_r)

            # Sinergym HourlyLinearReward / "schedule" reward (metric #3)
            hlr_reward, _ = compute_hourly_linear_reward(obs)
            hlr_rewards.append(hlr_reward)
            monthly_hlr_rewards.append(hlr_reward)

            # Real-world $ cost (info carries cumulative; take per-step deltas)
            ce = info.get('cost_energy_usd', 0.0)
            cd = info.get('cost_demand_usd', 0.0)
            monthly_cost_energy += ce - prev_ce; prev_ce = ce
            monthly_cost_demand += cd - prev_cd; prev_cd = cd

            power = obs[ObsIndex.HVAC_DEMAND]
            if floor_power_idx is not None:
                floor_power_rows.append([float(obs[i]) * floor_power_scale
                                         for i in floor_power_idx])
            energy_list.append(power)
            monthly_energy += power

            zok, ztotal = compute_zone_comfort_rate(obs, occupied_only=True)
            total_zones_ok += zok
            total_zone_checks += ztotal
            monthly_zones_ok += zok
            monthly_zone_checks += ztotal

            dev_val, dev_count = compute_mean_deviation(obs, occupied_only=True)
            total_deviation += dev_val
            total_dev_steps += dev_count
            monthly_deviation += dev_val
            monthly_dev_steps += dev_count

            cs_ep.update(obs)
            cs_m.update(obs)

            steps += 1
            monthly_steps += 1

            month_now = int(obs[ObsIndex.MONTH])
            if month_now != current_month:
                if current_month > 0:
                    energy_kwh = monthly_energy * 0.25 / 1000
                    avg_temp = agent.get_average_zone_temperature(obs)
                    zt = agent.get_zone_temperatures(obs)
                    zok_now = count_zones_ok(obs)
                    season = "S" if get_seasonal_comfort(current_month, 15) == COMFORT_SUMMER else "W"
                    m_zcr = (monthly_zones_ok / monthly_zone_checks * 100) if monthly_zone_checks > 0 else 100.0
                    m_dev = monthly_deviation / monthly_dev_steps if monthly_dev_steps > 0 else 0.0
                    m_cost = monthly_cost_energy + monthly_cost_demand
                    m_cs = cs_m.mean; m_cso = cs_m.mean_occ
                    print(f"  Month {current_month:2d} [{season}] | "
                          f"LinR: {sum(monthly_rewards):9.1f} | "
                          f"CS: {m_cs:.3f}/{m_cso:.3f} | "
                          f"CustomR: {sum(monthly_custom_rewards):8.2f} | "
                          f"HLR: {sum(monthly_hlr_rewards):9.1f} | "
                          f"Energy: {energy_kwh:7.0f} kWh | "
                          f"Cost: ${m_cost:8,.0f} (en ${monthly_cost_energy:,.0f}/dem ${monthly_cost_demand:,.0f}) | "
                          f"ZCR: {m_zcr:5.1f}% | "
                          f"Dev: {m_dev:.3f}°C | "
                          f"Avg: {avg_temp:5.1f}°C [{np.min(zt):.1f}-{np.max(zt):.1f}]")
                current_month = month_now
                monthly_rewards = []
                monthly_custom_rewards = []
                monthly_hlr_rewards = []
                monthly_energy = 0.0
                monthly_cost_energy = 0.0
                monthly_cost_demand = 0.0
                monthly_zones_ok = 0
                monthly_zone_checks = 0
                monthly_deviation = 0.0
                monthly_dev_steps = 0
                monthly_steps = 0
                cs_m.reset()

        total_energy_kwh = sum(energy_list) * 0.25 / 1000
        zcr = (total_zones_ok / total_zone_checks * 100) if total_zone_checks > 0 else 100.0
        mean_dev = total_deviation / total_dev_steps if total_dev_steps > 0 else 0.0
        ep_cost_energy = info.get('cost_energy_usd', 0.0)
        ep_cost_demand = info.get('cost_demand_usd', 0.0)
        ep_cost_total  = info.get('cost_total_usd', ep_cost_energy + ep_cost_demand)
        ep_cs = cs_ep.mean; ep_cs_occ = cs_ep.mean_occ

        all_rewards.extend(rewards)
        all_custom_rewards.append(sum(custom_rewards))
        all_hlr_rewards.append(sum(hlr_rewards))
        all_energy.append(total_energy_kwh)
        all_zcr.append(zcr)
        all_dev.append(mean_dev)
        all_cost.append(ep_cost_total)
        all_cs.append(ep_cs); all_cs_occ.append(ep_cs_occ)

        print("-" * 80)
        print(f"  Episode {ep+1} Summary:")
        print(f"    Linear Reward (env):   {sum(rewards):.2f}   [Sinergym LinearReward]")
        print(f"    Custom Reward:         {sum(custom_rewards):.2f}   [weighted custom]")
        print(f"    Hourly Reward (sched): {sum(hlr_rewards):.2f}   [Sinergym HourlyLinearReward]")
        print(f"    Total Energy:          {total_energy_kwh:,.0f} kWh")
        _peak_kW = (max(energy_list) / 1000.0) if energy_list else float('nan')
        _lf = load_factor(energy_list)
        print(f"    Peak Demand:           {_peak_kW:,.1f} kW")
        print(f"    Load Factor:           {_lf:.3f}   (avg/peak; higher=flatter=cheaper)")
        if floor_power_idx is not None and len(floor_power_rows) > 0:
            _fp = np.asarray(floor_power_rows, dtype=np.float64)
            _cf = floor_coincidence_factor(_fp)
            _fpeaks = _fp.max(axis=0) / 1000.0
            _coinc = _fp.sum(axis=1).max() / 1000.0
            print(f"    Floor Peaks (kW):      bot={_fpeaks[0]:.1f}  mid={_fpeaks[1]:.1f}  "
                  f"top={_fpeaks[2]:.1f}  (sum individual={_fpeaks.sum():.1f})")
            print(f"    Coincident Peak:       {_coinc:.1f} kW")
            print(f"    Coincidence Factor:    {_cf:.3f}   "
                  f"(1=floors peak together/expensive, lower=staggered/cheaper)")
        print(f"    Energy Cost:           ${ep_cost_energy:,.2f}")
        print(f"    Demand Cost:           ${ep_cost_demand:,.2f}")
        print(f"    Total Cost:            ${ep_cost_total:,.2f}")
        print(f"    CS (weighted/occ):     {ep_cs:.4f} / {ep_cs_occ:.4f}")
        print(f"    Zone-Comfort Rate:     {zcr:.1f}%  (occupied hours only)")
        print(f"    Mean Deviation:        {mean_dev:.3f} °C  (occupied hours only)")
        print()

    env.close()

    print(f"{'='*80}")
    print(f"RBC FINAL SUMMARY  [{granularity} | {struct_label}]")
    print(f"{'='*80}")
    print(f"  Episodes:              {episodes}")
    print(f"  Linear Reward (env):   {sum(all_rewards):.2f}   [Sinergym LinearReward]")
    print(f"  Custom Reward total:   {sum(all_custom_rewards):.2f}   [weighted custom]")
    print(f"  Hourly Reward (sched): {sum(all_hlr_rewards):.2f}   [Sinergym HourlyLinearReward]")
    print(f"  Total Energy:          {sum(all_energy):,.0f} kWh")
    print(f"  Total Cost:            ${sum(all_cost):,.2f}")
    if episodes > 1:
        print(f"  Cost per episode:      ${np.mean(all_cost):,.2f} (mean)")
    print(f"  CS (weighted/occ):     {np.nanmean(all_cs):.4f} / {np.nanmean(all_cs_occ):.4f}")
    print(f"  Zone-Comfort Rate:     {np.mean(all_zcr):.1f}%  (occupied hours only)")
    print(f"  Mean Deviation:        {np.mean(all_dev):.3f} °C  (occupied hours only)")
    print(f"{'='*80}")

    return {
        'linear_rewards':       all_rewards,
        'custom_reward_total':  sum(all_custom_rewards),
        'hourly_reward_total':  sum(all_hlr_rewards),
        'per_episode_custom':   all_custom_rewards,
        'per_episode_hlr':      all_hlr_rewards,
        'total_energy_kwh':     sum(all_energy),
        'total_cost_usd':       sum(all_cost),
        'per_episode_cost':     all_cost,
        'zone_comfort_rate':    np.mean(all_zcr),
        'mean_deviation':       np.mean(all_dev),
        'comfort_score':        float(np.nanmean(all_cs)),
        'comfort_score_occ':    float(np.nanmean(all_cs_occ)),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RBC Unified Controller")
    parser.add_argument("--granularity", type=str, default="zone",
                        choices=["zone", "floor", "single"],
                        help="Control granularity: zone, floor, or single")
    parser.add_argument("--structural", action="store_true", default=True,
                        help="Enable structural dependencies (default: True)")
    parser.add_argument("--no-structural", dest="structural", action="store_false",
                        help="Disable structural dependencies")
    parser.add_argument("--weather", type=str, default="mixed")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--cool-margin", type=float, default=1.0, dest="cool_margin",
                        help="Cooling held at comfort_low + margin (summer: 23+margin). "
                             "Default 1.0 -> 24C in summer.")
    parser.add_argument("--floor_power_idx", type=str, default=None,
                        help="Comma-separated obs indices of per-floor HVAC power "
                             "[bot,mid,top] (e.g. '92,93,94'). Enables coincidence metric.")
    parser.add_argument("--floor_power_scale", type=float, default=1.0,
                        help="Per-floor obs multiplier; meters report J/timestep so use "
                             "0.0011111 (=1/900s) for kW. CF is scale-invariant.")
    args = parser.parse_args()

    results = run_simulation(
        weather=args.weather,
        episodes=args.episodes,
        granularity=args.granularity,
        structural=args.structural,
        cool_margin=args.cool_margin,
        floor_power_idx=args.floor_power_idx,
        floor_power_scale=args.floor_power_scale,
    )