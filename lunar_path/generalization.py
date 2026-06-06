from dataclasses import replace
from pathlib import Path
from typing import List

import pandas as pd
from stable_baselines3 import DQN, PPO

from lunar_path.environment import execute_path_with_battery, generate_lunar_map
from lunar_path.metrics import path_metrics
from lunar_path.methods.deep_rl import deep_rl_policy_path
from lunar_path.scenarios import default_scenarios


def run_generalization_test(model_dir: Path, out_path: Path, seed_offset: int = 9000) -> pd.DataFrame:
    rows = []
    for scenario in default_scenarios():
        unseen = replace(scenario, seed=scenario.seed + seed_offset, title=f"{scenario.title} Unseen Seed")
        lunar = generate_lunar_map(unseen)
        for method_name, model_cls in [("DQN", DQN), ("PPO", PPO)]:
            model = model_cls.load(model_dir / f"{scenario.name}_{method_name.lower()}.zip", device="cuda")
            path = execute_path_with_battery(lunar, deep_rl_policy_path(lunar, model))
            row = path_metrics(lunar, path, f"{method_name} unseen-map", unseen)
            row["train_scenario"] = scenario.name
            row["test_scenario"] = unseen.name
            row["train_seed"] = scenario.seed
            row["test_seed"] = unseen.seed
            rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(out_path, index=False)
    return results

