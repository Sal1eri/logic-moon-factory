from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lunar_path.environment import Coord, LunarMap, Scenario


METHOD_ORDER = [
    "Random",
    "DFS",
    "A* shortest",
    "BFS",
    "Greedy",
    "DQN",
    "PPO",
    "A* risk-aware",
]

SCENARIO_LABELS = {
    "baseline": "baseline",
    "complex_moon": "complex",
    "crater_field": "crater",
    "low_battery_bad_case": "battery_stress",
    "shadow_comm": "shadow_comm",
    "slope_ridges": "ridges",
}


def ordered_methods(methods):
    base = {method: idx for idx, method in enumerate(METHOD_ORDER)}

    def key(method):
        normalized = method.replace("-online", "")
        return (base.get(normalized, 999), method)

    return sorted(methods, key=key)


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
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = ordered_methods(list(paths.keys()))
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
    cols = 4
    rows = int(np.ceil(len(methods) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.8 * rows), constrained_layout=True)
    axes_flat = list(np.atleast_1d(axes).flat)
    for ax, method in zip(axes_flat, methods):
        ax.imshow(base_rgb(lunar))
        path = paths.get(method)
        if path:
            arr = np.array(path)
            ax.plot(arr[:, 1], arr[:, 0], color=colors.get(method, "white"), lw=2.3)
            finished = path[-1] == lunar.goal
            marker = "o" if finished else "x"
            size = 70 if finished else 85
            end_x = float(arr[-1, 1])
            end_y = float(arr[-1, 0])
            if finished:
                end_x += 0.48
                end_y += 0.48
            ax.scatter([end_x], [end_y], c=colors.get(method, "white"), s=size, marker=marker, linewidths=2.2, edgecolor="black" if finished else None, zorder=8)
            status = "PASS" if finished else "FAIL"
        else:
            status = "NO PATH"
        ax.scatter([lunar.start[1]], [lunar.start[0]], c="#00e676", s=60, marker="o", edgecolor="black", zorder=10)
        ax.scatter([lunar.goal[1]], [lunar.goal[0]], c="white", s=170, marker="*", edgecolor="black", linewidths=1.1, zorder=11)
        ax.scatter([lunar.goal[1]], [lunar.goal[0]], c="#ff1744", s=110, marker="*", edgecolor="black", linewidths=0.8, zorder=12)
        ax.set_title(f"{method} - {status}", fontsize=10, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_flat[len(methods):]:
        ax.axis("off")
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
    g.savefig(out_dir / "rl_training_reward.png", dpi=180, bbox_inches="tight")
    plt.close(g.fig)

    episode_counts = logs.groupby(["scenario", "algorithm"]).size().reset_index(name="completed_episodes")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=episode_counts, x="scenario", y="completed_episodes", hue="algorithm", ax=ax, palette="Set2")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Episodes")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(out_dir / "rl_training_episodes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(metrics: pd.DataFrame, out_dir: Path) -> None:
    success = metrics[metrics["path_found"] == 1].copy()
    metric_names = ["path_length", "total_cost", "energy", "terrain_risk", "shadow_ratio", "comm_blackout_ratio"]
    compact = success[["scenario", "method"] + metric_names].copy()
    for scenario, group in compact.groupby("scenario"):
        for metric in metric_names:
            values = group[metric]
            min_v = values.min()
            max_v = values.max()
            compact.loc[group.index, metric] = (values - min_v) / (max_v - min_v + 1e-9)

    scenarios = list(compact["scenario"].drop_duplicates())
    cols = 3
    rows = int(np.ceil(len(scenarios) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14.5, 3.6 * rows), constrained_layout=True)
    axes_flat = list(np.atleast_1d(axes).flat)
    for ax, scenario in zip(axes_flat, scenarios):
        data = compact[compact["scenario"] == scenario].set_index("method")[metric_names]
        data = data.reindex([m for m in ordered_methods(data.index) if m in data.index])
        labels = data.copy()
        labels.columns = [c.replace("_", "\n") for c in labels.columns]
        sns.heatmap(
            labels,
            annot=True,
            fmt=".2f",
            cmap="magma_r",
            vmin=0,
            vmax=1,
            linewidths=0.45,
            linecolor="white",
            cbar=False,
            ax=ax,
        )
        ax.set_title(scenario, fontsize=10, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    for ax in axes_flat[len(scenarios):]:
        ax.axis("off")
    fig.savefig(out_dir / "metrics_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    pivot = metrics.pivot_table(index=["scenario", "method"], values=metric_names, aggfunc="mean")
    norm = pivot.copy()
    for col in metric_names:
        values = norm[col].replace([np.inf, -np.inf], np.nan)
        norm[col] = (values - values.min()) / (values.max() - values.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(9, 10))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="magma_r", ax=ax, cbar_kws={"label": "normalized lower-is-better"})
    fig.savefig(out_dir / "metrics_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=metrics, x="scenario", y="task_success", hue="method", ax=ax, palette="Set2")
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
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Battery capacity - path energy")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(out_dir / "battery_energy_margin.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    outcome = metrics.pivot(index="method", columns="scenario", values="task_success")
    outcome = outcome.reindex([m for m in ordered_methods(outcome.index) if m in outcome.index])
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
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Method")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(out_dir / "task_outcome_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_offline_online_outcomes(metrics: pd.DataFrame, online: pd.DataFrame, out_dir: Path) -> None:
    if metrics.empty or online.empty:
        return
    online = online[online["method"] != "D* Lite-style"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)
    datasets = [
        ("Offline full-map", metrics, "method"),
        ("Online local-view", online, "method"),
    ]
    for ax, (label, data, method_col) in zip(axes, datasets):
        outcome = data.pivot(index=method_col, columns="scenario", values="task_success")
        outcome = outcome.reindex([m for m in ordered_methods(outcome.index) if m in outcome.index])
        outcome = outcome.rename(columns=SCENARIO_LABELS)
        labels = outcome.map(lambda value: "PASS" if value == 1 else "FAIL")
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
        ax.set_title(label, fontsize=11, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25)
        ax.tick_params(axis="y", labelsize=8)
    fig.savefig(out_dir / "offline_online_outcome_matrix.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_generalization(generalization: pd.DataFrame, out_dir: Path) -> None:
    if generalization.empty:
        return
    outcome = generalization.pivot(index="method", columns="train_scenario", values="task_success")
    outcome = outcome.reindex([m for m in ordered_methods(outcome.index) if m in outcome.index])
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
    ax.set_xlabel("Training scenario tested on unseen seed")
    ax.set_ylabel("Saved RL model")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(out_dir / "rl_generalization_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
