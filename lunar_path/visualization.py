from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lunar_path.environment import Coord, LunarMap, Scenario


def base_rgb(lunar: LunarMap) -> np.ndarray:
    n = lunar.size
    rgb = np.zeros((n, n, 3), dtype=float)
    elev = (lunar.elevation - np.nanmin(lunar.elevation)) / (np.nanmax(lunar.elevation) - np.nanmin(lunar.elevation) + 1e-9)
    rgb[..., 0] = 0.36 + 0.34 * elev
    rgb[..., 1] = 0.35 + 0.32 * elev
    rgb[..., 2] = 0.33 + 0.30 * elev
    rgb[lunar.regolith > 0.25] *= np.array([1.08, 0.95, 0.78])
    rgb[lunar.crater_risk > 0.4] *= np.array([0.72, 0.72, 0.75])
    rgb[lunar.illumination < 0.35] *= np.array([0.30, 0.35, 0.50])
    rgb[lunar.communication < 0.20] = rgb[lunar.communication < 0.20] * 0.86 + np.array([0.12, 0.02, 0.12])
    rgb[lunar.obstacle] = np.array([0.04, 0.04, 0.045])
    return np.clip(rgb, 0, 1)


def plot_environment(lunar: LunarMap, scenario: Scenario, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fields = [
        ("Moon-like terrain", base_rgb(lunar), None),
        ("Elevation", lunar.elevation, "mako"),
        ("Slope risk", lunar.slope, "flare"),
        ("Crater / rock risk", lunar.crater_risk + lunar.obstacle.astype(float), "rocket_r"),
        ("Illumination", lunar.illumination, "viridis"),
        ("Communication", lunar.communication, "crest"),
    ]
    for ax, (title, data, cmap) in zip(axes.flat, fields):
        if cmap is None:
            ax.imshow(data)
        else:
            im = ax.imshow(data, cmap=cmap)
            fig.colorbar(im, ax=ax, shrink=0.78)
        ax.scatter([lunar.start[1]], [lunar.start[0]], c="#00e676", s=80, marker="o", edgecolor="black")
        ax.scatter([lunar.goal[1]], [lunar.goal[0]], c="#ff1744", s=95, marker="*", edgecolor="black")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Lunar Scenario: {scenario.title}", fontsize=16, weight="bold")
    fig.savefig(out_dir / f"{scenario.name}_environment.png", dpi=180)
    plt.close(fig)


def plot_paths(lunar: LunarMap, scenario: Scenario, paths: Dict[str, Optional[List[Coord]]], out_dir: Path) -> None:
    colors = {
        "Random": "#9e9e9e",
        "DFS": "#795548",
        "BFS": "#3f51b5",
        "Greedy": "#ff9800",
        "A* shortest": "#00bcd4",
        "A* risk-aware": "#76ff03",
        "DQN": "#ffeb3b",
        "PPO": "#ff4081",
    }
    fig, ax = plt.subplots(figsize=(10.5, 8))
    ax.imshow(base_rgb(lunar))
    edge_users = {}
    for method, path in paths.items():
        if not path or len(path) < 2:
            continue
        for a, b in zip(path[:-1], path[1:]):
            edge = tuple(sorted((a, b)))
            edge_users.setdefault(edge, []).append(method)

    for method, path in paths.items():
        if not path:
            continue
        arr = np.array(path)
        xs = arr[:, 1].astype(float)
        ys = arr[:, 0].astype(float)
        x_plot = [xs[0]]
        y_plot = [ys[0]]
        for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
            edge = tuple(sorted((a, b)))
            users = edge_users.get(edge, [method])
            if len(users) > 1:
                users_sorted = sorted(users)
                rank = users_sorted.index(method)
                centered_rank = rank - (len(users_sorted) - 1) / 2.0
                dx = b[1] - a[1]
                dy = b[0] - a[0]
                norm = max((dx * dx + dy * dy) ** 0.5, 1e-9)
                offset = 0.12 * centered_rank
                ox = -dy / norm * offset
                oy = dx / norm * offset
            else:
                ox = 0.0
                oy = 0.0
            x_plot.append(xs[i + 1] + ox)
            y_plot.append(ys[i + 1] + oy)
        ax.plot(
            x_plot,
            y_plot,
            color=colors.get(method, "white"),
            lw=2.1,
            alpha=0.92,
            label=method,
            marker="o",
            markersize=2.5,
            markevery=max(1, len(path) // 8),
        )
    ax.scatter([lunar.start[1]], [lunar.start[0]], c="#00e676", s=105, marker="o", edgecolor="black", label="Start")
    ax.scatter([lunar.goal[1]], [lunar.goal[0]], c="#ff1744", s=140, marker="*", edgecolor="black", label="Goal")
    ax.set_title(f"Path Comparison - {scenario.title}", fontsize=14, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.92,
        fontsize=8,
        borderaxespad=0.0,
    )
    fig.savefig(out_dir / f"{scenario.name}_paths.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_path_panels(lunar: LunarMap, scenario: Scenario, paths: Dict[str, Optional[List[Coord]]], out_dir: Path) -> None:
    methods = list(paths.keys())
    colors = {
        "Random": "#9e9e9e",
        "DFS": "#795548",
        "BFS": "#3f51b5",
        "Greedy": "#ff9800",
        "A* shortest": "#00bcd4",
        "A* risk-aware": "#76ff03",
        "DQN": "#fdd835",
        "PPO": "#ff4081",
    }
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), constrained_layout=True)
    for ax, method in zip(axes.flat, methods):
        ax.imshow(base_rgb(lunar))
        path = paths.get(method)
        if path:
            arr = np.array(path)
            ax.plot(arr[:, 1], arr[:, 0], color=colors.get(method, "white"), lw=2.3)
            finished = path[-1] == lunar.goal
            marker = "o" if finished else "x"
            size = 70 if finished else 85
            ax.scatter([arr[-1, 1]], [arr[-1, 0]], c=colors.get(method, "white"), s=size, marker=marker, linewidths=2.2, edgecolor="black" if finished else None)
            status = "PASS" if finished else "FAIL"
        else:
            status = "NO PATH"
        ax.scatter([lunar.start[1]], [lunar.start[0]], c="#00e676", s=55, marker="o", edgecolor="black")
        ax.scatter([lunar.goal[1]], [lunar.goal[0]], c="#ff1744", s=78, marker="*", edgecolor="black")
        ax.set_title(f"{method} - {status}", fontsize=10, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(methods):]:
        ax.axis("off")
    fig.suptitle(f"Individual Method Paths - {scenario.title}", fontsize=16, weight="bold")
    fig.savefig(out_dir / f"{scenario.name}_path_panels.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_training(logs: pd.DataFrame, out_dir: Path) -> None:
    if logs.empty:
        return
    logs = logs.copy()
    logs["reward_ma"] = logs.groupby(["scenario", "algorithm"])["reward"].transform(lambda s: s.rolling(60, min_periods=1).mean())
    g = sns.relplot(
        data=logs,
        x="episode",
        y="reward_ma",
        hue="algorithm",
        col="scenario",
        col_wrap=3,
        kind="line",
        height=3.0,
        aspect=1.25,
        facet_kws={"sharey": False},
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Episode", "Smoothed reward")
    g.fig.suptitle("Deep RL Training Reward Curves", y=1.03, fontsize=15, weight="bold")
    g.savefig(out_dir / "rl_training_reward.png", dpi=180, bbox_inches="tight")
    plt.close(g.fig)

    episode_counts = logs.groupby(["scenario", "algorithm"]).size().reset_index(name="completed_episodes")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=episode_counts, x="scenario", y="completed_episodes", hue="algorithm", ax=ax, palette="Set2")
    ax.set_title("Completed Training Episodes", fontsize=14, weight="bold")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Episodes")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(out_dir / "rl_training_episodes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(metrics: pd.DataFrame, out_dir: Path) -> None:
    success = metrics[metrics["path_found"] == 1].copy()
    metric_names = ["path_length", "total_cost", "energy", "terrain_risk", "shadow_ratio", "comm_blackout_ratio"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for ax, metric in zip(axes.flat, metric_names):
        sns.barplot(data=success, x="scenario", y=metric, hue="method", ax=ax, palette="Set2")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        if ax.get_legend() is not None:
            ax.legend_.remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Path Planning Metrics Across Lunar Scenarios", fontsize=16, weight="bold")
    fig.savefig(out_dir / "metrics_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pivot = metrics.pivot_table(index=["scenario", "method"], values=metric_names, aggfunc="mean")
    norm = pivot.copy()
    for col in metric_names:
        values = norm[col].replace([np.inf, -np.inf], np.nan)
        norm[col] = (values - values.min()) / (values.max() - values.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(9, 10))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="magma_r", ax=ax, cbar_kws={"label": "normalized lower-is-better"})
    ax.set_title("Normalized Metric Heatmap", fontsize=14, weight="bold")
    fig.savefig(out_dir / "metrics_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=metrics, x="scenario", y="task_success", hue="method", ax=ax, palette="Set2")
    ax.set_title("Task Success Under Battery Constraint", fontsize=14, weight="bold")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Task success = path found and energy feasible")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, 1.05)
    fig.savefig(out_dir / "battery_task_success.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    feasible_paths = metrics[metrics["path_found"] == 1].copy()
    sns.barplot(data=feasible_paths, x="scenario", y="energy_margin", hue="method", ax=ax, palette="Set2")
    ax.axhline(0, color="black", lw=1.2, linestyle="--")
    ax.set_title("Battery Energy Margin", fontsize=14, weight="bold")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Battery capacity - path energy")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(out_dir / "battery_energy_margin.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    outcome = metrics.pivot(index="method", columns="scenario", values="task_success")
    labels = outcome.map(lambda value: "PASS" if value == 1 else "FAIL")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.heatmap(
        outcome,
        annot=labels,
        fmt="",
        cmap=sns.color_palette(["#d73027", "#1a9850"], as_cmap=True),
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=0.6,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Task Outcome Matrix", fontsize=14, weight="bold")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Method")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(out_dir / "task_outcome_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_generalization(generalization: pd.DataFrame, out_dir: Path) -> None:
    if generalization.empty:
        return
    outcome = generalization.pivot(index="method", columns="train_scenario", values="task_success")
    labels = outcome.map(lambda value: "PASS" if value == 1 else "FAIL")
    fig, ax = plt.subplots(figsize=(10, 3.2))
    sns.heatmap(
        outcome,
        annot=labels,
        fmt="",
        cmap=sns.color_palette(["#d73027", "#1a9850"], as_cmap=True),
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=0.6,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("DQN/PPO Unseen-Map Generalization", fontsize=14, weight="bold")
    ax.set_xlabel("Training scenario tested on unseen seed")
    ax.set_ylabel("Saved RL model")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(out_dir / "rl_generalization_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
