from typing import Dict, List, Optional

import numpy as np

from lunar_path.environment import Coord, LunarMap, Scenario, transition_cost, transition_energy


def path_metrics(lunar: LunarMap, path: Optional[List[Coord]], method: str, scenario: Scenario) -> Dict[str, float]:
    row = {
        "scenario": scenario.name,
        "scenario_title": scenario.title,
        "method": method,
        "path_found": 0,
        "energy_feasible": 0,
        "task_success": 0,
        "battery_capacity": scenario.battery_capacity,
        "energy_margin": np.nan,
        "path_length": np.nan,
        "total_cost": np.nan,
        "energy": np.nan,
        "terrain_risk": np.nan,
        "shadow_ratio": np.nan,
        "comm_blackout_ratio": np.nan,
        "avg_slope": np.nan,
    }
    if not path:
        return row
    arr = np.array(path)
    diffs = np.diff(arr, axis=0)
    step_dist = np.sqrt(np.sum(diffs * diffs, axis=1)) if len(path) > 1 else np.array([])
    dist = float(np.sum(step_dist)) if len(path) > 1 else 0.0
    cells = (arr[:, 0], arr[:, 1])
    total_cost = 0.0
    energy = 0.0
    if len(path) > 1:
        for idx, move in enumerate(step_dist):
            src = tuple(arr[idx])
            dst = tuple(arr[idx + 1])
            total_cost += transition_cost(lunar, src, dst, float(move))
            energy += transition_energy(lunar, src, dst, float(move))
    path_found = int(path[-1] == lunar.goal)
    energy_feasible = int(energy <= scenario.battery_capacity)
    row.update({
        "path_found": path_found,
        "energy_feasible": energy_feasible,
        "task_success": int(path_found and energy_feasible),
        "energy_margin": float(scenario.battery_capacity - energy),
        "path_length": dist,
        "total_cost": total_cost,
        "energy": energy,
        "terrain_risk": float(np.sum(lunar.crater_risk[cells] + lunar.slope[cells])),
        "shadow_ratio": float(np.mean(lunar.illumination[cells] < 0.35)),
        "comm_blackout_ratio": float(np.mean(lunar.communication[cells] < 0.25)),
        "avg_slope": float(np.mean(lunar.slope[cells])),
    })
    return row
