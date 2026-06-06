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
from lunar_path.methods.deep_rl import deep_rl_policy_path, train_deep_rl
from lunar_path.methods.dfs import depth_first_search
from lunar_path.methods.greedy import greedy_best_first
from lunar_path.methods.random_planner import random_walk
from lunar_path.report import write_report
from lunar_path.scenarios import default_scenarios
from lunar_path.visualization import plot_environment, plot_metric_comparison, plot_path_panels, plot_paths, plot_training


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
    all_logs = []

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

        for method_name, seed_offset in [("DQN", 1000), ("PPO", 2000)]:
            print(f"  training {method_name} for {scenario.rl_timesteps} timesteps", flush=True)
            model, logs = train_deep_rl(lunar, method_name, scenario.rl_timesteps, scenario.seed + seed_offset, device)
            model.save(model_dir / f"{scenario.name}_{method_name.lower()}")
            logs.insert(0, "scenario", scenario.name)
            all_logs.append(logs)
            paths[method_name] = execute_path_with_battery(lunar, deep_rl_policy_path(lunar, model))

        plot_paths(lunar, scenario, paths, fig_dir)
        plot_path_panels(lunar, scenario, paths, fig_dir)
        for method, path in paths.items():
            all_metrics.append(path_metrics(lunar, path, method, scenario))

    metrics = pd.DataFrame(all_metrics)
    logs = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    logs.to_csv(out_dir / "rl_training_logs.csv", index=False)
    plot_metric_comparison(metrics, fig_dir)
    plot_training(logs, fig_dir)
    write_report(metrics, scenarios, root / "report.md")

    print("Done.", flush=True)
    print(f"Report: {root / 'report.md'}", flush=True)
    print(f"Metrics: {out_dir / 'metrics.csv'}", flush=True)
    print(f"Figures: {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
