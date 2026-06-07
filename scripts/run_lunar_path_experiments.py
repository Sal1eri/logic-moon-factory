import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import seaborn as sns
import torch

from lunar_path.environment import execute_path_with_battery, generate_lunar_map
from lunar_path.metrics import path_metrics
from lunar_path.methods.astar import astar
from lunar_path.methods.bfs import breadth_first_search
from stable_baselines3 import DQN, PPO

from lunar_path.methods.deep_rl import deep_rl_policy_path
from lunar_path.methods.dfs import depth_first_search
from lunar_path.methods.greedy import greedy_best_first
from lunar_path.methods.random_planner import random_walk
from lunar_path.report import write_report
from lunar_path.scenarios import default_scenarios
from lunar_path.visualization import plot_environment, plot_metric_comparison, plot_path_panels, plot_paths


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=0.92)
    root = Path("experiments")
    out_dir = root / "results"
    fig_dir = out_dir / "figures"
    model_dir = out_dir / "models"
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}", flush=True)

    scenarios = default_scenarios()
    all_metrics = []
    rl_models = {
        "DQN": DQN.load(model_dir / "global_dqn_randomized.zip", device=device),
        "PPO": PPO.load(model_dir / "global_ppo_randomized.zip", device=device),
    }

    for scenario in scenarios:
        print(f"Running scenario: {scenario.name}", flush=True)
        lunar = generate_lunar_map(scenario)
        plot_environment(lunar, scenario, fig_dir)

        paths = {
            "Random": random_walk(lunar, seed=scenario.seed + 10),
            "DFS": depth_first_search(lunar),
            "BFS": breadth_first_search(lunar),
            "Greedy": greedy_best_first(lunar),
            "A* shortest": astar(lunar, risk_aware=False, battery_constrained=False),
            "A* risk-aware": astar(lunar, risk_aware=True, battery_constrained=True),
        }
        paths = {method: execute_path_with_battery(lunar, path) for method, path in paths.items()}

        for method_name, model in rl_models.items():
            paths[method_name] = execute_path_with_battery(lunar, deep_rl_policy_path(lunar, model))

        plot_paths(lunar, scenario, paths, fig_dir)
        plot_path_panels(lunar, scenario, paths, fig_dir)
        for method, path in paths.items():
            all_metrics.append(path_metrics(lunar, path, method, scenario))

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    plot_metric_comparison(metrics, fig_dir)
    write_report(metrics, scenarios, root / "report.md")

    print("Done.", flush=True)
    print(f"Report: {root / 'report.md'}", flush=True)
    print(f"Metrics: {out_dir / 'metrics.csv'}", flush=True)
    print(f"Figures: {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
