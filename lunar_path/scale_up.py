from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3 import DQN, PPO

from lunar_path.environment import execute_path_with_battery, generate_lunar_map
from lunar_path.metrics import path_metrics
from lunar_path.methods.astar import astar
from lunar_path.methods.bfs import breadth_first_search
from lunar_path.methods.deep_rl import deep_rl_policy_path
from lunar_path.methods.dfs import depth_first_search
from lunar_path.methods.greedy import greedy_best_first
from lunar_path.methods.random_planner import random_walk
from lunar_path.online import run_online_method, run_online_rl_method
from lunar_path.scenarios import default_scenarios


OFFLINE_METHODS = ["Random", "DFS", "BFS", "Greedy", "A* shortest", "A* risk-aware", "DQN", "PPO"]
ONLINE_METHODS = ["Random-online", "DFS-online", "BFS-online", "Greedy-online", "A* shortest-online", "A* risk-aware-online", "DQN-online", "PPO-online"]
SCENARIO_LABELS = {
    "baseline": "baseline",
    "complex_moon": "complex",
    "crater_field": "crater",
    "low_battery_bad_case": "battery_stress",
    "shadow_comm": "shadow_comm",
    "slope_ridges": "ridges",
}


def run_scale_up_experiment(
    out_dir: Path,
    model_dir: Path,
    seeds_per_scenario: int = 8,
    sensing_radius: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    rows = []
    rl_models = {
        "DQN": DQN.load(model_dir / "global_dqn_randomized.zip", device="cpu"),
        "PPO": PPO.load(model_dir / "global_ppo_randomized.zip", device="cpu"),
    }
    for scenario in default_scenarios():
        if verbose:
            print(f"scale-up scenario: {scenario.name}", flush=True)
        for seed_idx in range(seeds_per_scenario):
            if verbose and (seed_idx == 0 or (seed_idx + 1) % 5 == 0):
                print(f"  seed {seed_idx + 1}/{seeds_per_scenario}", flush=True)
            test_seed = scenario.seed + 10000 + seed_idx * 37
            sampled = replace(scenario, seed=test_seed)
            lunar = generate_lunar_map(sampled)

            offline_paths = {
                "Random": random_walk(lunar, seed=test_seed + 13),
                "DFS": depth_first_search(lunar),
                "BFS": breadth_first_search(lunar),
                "Greedy": greedy_best_first(lunar),
                "A* shortest": astar(lunar, risk_aware=False, battery_constrained=False),
                "A* risk-aware": astar(lunar, risk_aware=True, battery_constrained=True),
            }
            offline_paths = {method: execute_path_with_battery(lunar, path) for method, path in offline_paths.items()}

            for method_name, model in rl_models.items():
                offline_paths[method_name] = execute_path_with_battery(lunar, deep_rl_policy_path(lunar, model))

            for method, path in offline_paths.items():
                row = path_metrics(lunar, path, method, sampled)
                row.update({"setting": "offline_full_map", "sample_seed": test_seed})
                rows.append(row)

            for method in ["Random-online", "DFS-online", "BFS-online", "Greedy-online", "A* shortest-online", "A* risk-aware-online"]:
                path, info = run_online_method(lunar, sampled, method, sensing_radius=sensing_radius, seed=test_seed + 77)
                row = path_metrics(lunar, path, method, sampled)
                row.update(info)
                row.update({"setting": "online_local_view", "sample_seed": test_seed})
                rows.append(row)

            for method_name, model in [("DQN-online", rl_models["DQN"]), ("PPO-online", rl_models["PPO"])]:
                path, info = run_online_rl_method(lunar, sampled, model, sensing_radius=sensing_radius)
                row = path_metrics(lunar, path, method_name, sampled)
                row.update(info)
                row.update({"setting": "online_local_view", "sample_seed": test_seed})
                rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "scale_up_metrics.csv", index=False)
    plot_scale_up_results(results, out_dir / "figures")
    return results


def plot_scale_up_results(results: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        results.groupby(["setting", "scenario", "method"])
        .agg(
            pass_rate=("task_success", "mean"),
            mean_energy=("energy", "mean"),
            mean_path_length=("path_length", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(fig_dir.parent / "scale_up_summary.csv", index=False)

    for setting in sorted(summary["setting"].unique()):
        data = summary[summary["setting"] == setting]
        pivot = data.pivot(index="method", columns="scenario", values="pass_rate")
        method_order = pivot.mean(axis=1).sort_values(ascending=True).index
        pivot = pivot.reindex(method_order)
        pivot = pivot.rename(columns=SCENARIO_LABELS)
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlGn",
            vmin=0,
            vmax=1,
            linewidths=0.6,
            linecolor="white",
            ax=ax,
            cbar_kws={"label": ""},
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25)
        fig.savefig(fig_dir / f"scale_up_{setting}_pass_rate.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    overall_order = (
        summary.groupby("method")["pass_rate"]
        .mean()
        .sort_values(ascending=True)
        .index
    )
    sns.barplot(data=summary, x="method", y="pass_rate", hue="setting", order=overall_order, ax=ax, palette="Set2")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(fig_dir / "scale_up_overall_pass_rate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
