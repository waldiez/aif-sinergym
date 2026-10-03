# Building model

Place `OfficeMedium_MultiAgent_perfloor.epJSON` in this folder. It is the
ASHRAE 90.1 OfficeMedium prototype (Denver) with:

* per-zone heating and cooling setpoint schedules exposed as actuators, and
* three custom electricity meters, one per air loop (fan, cooling coil and the
  reheat coils of that floor), read by the agents as `floor_power_bot/mid/top`.

Sinergym looks for building files in its own `data/buildings` folder. After
adding the file here, install it once with

    python scripts/install_building.py
