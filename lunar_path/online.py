import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3 import DQN, PPO

from lunar_path.environment import (
    ACTIONS,
    Coord,
    LunarMap,
    Scenario,
    execute_path_with_battery,
    generate_lunar_map,
    heuristic,
    transition_energy,
)
from lunar_path.metrics import path_metrics
from lunar_path.methods.astar import astar
from lunar_path.methods.bfs import breadth_first_search
from lunar_path.methods.deep_rl import deep_rl_policy_path
from lunar_path.methods.dfs import depth_first_search
from lunar_path.methods.greedy import greedy_best_first
from lunar_path.methods.random_planner import random_walk
from lunar_path.report import df_to_markdown
from lunar_path.scenarios import default_scenarios
from lunar_path.visualization import base_rgb, plot_path_panels


def make_belief_map(true_map: LunarMap, known: np.ndarray) -> LunarMap:
    n = true_map.size
    elevation = np.where(known, true_map.elevation, true_map.elevation[true_map.start])
    slope = np.where(known, true_map.slope, 0.18)
    obstacle = np.where(known, true_map.obstacle, False)
    crater_risk = np.where(known, true_map.crater_risk, 0.05)
    regolith = np.where(known, true_map.regolith, 0.12)
    illumination = np.where(known, true_map.illumination, 0.75)
    communication = np.where(known, true_map.communication, 0.55)
    cost = (
        1.0
        + 2.8 * slope
        + 2.2 * crater_risk
        + 1.7 * regolith
        + 1.6 * (1.0 - illumination)
        + 1.3 * (1.0 - communication)
    )
    cost[obstacle] = np.inf
    return LunarMap(
        n,
        elevation,
        slope,
        obstacle,
        crater_risk,
        regolith,
        illumination,
        communication,
        cost,
        true_map.start,
        true_map.goal,
        true_map.battery_capacity,
    )


def sense(true_map: LunarMap, known: np.ndarray, pos: Coord, radius: int) -> None:
    yy, xx = np.indices((true_map.size, true_map.size))
    mask = (yy - pos[0]) ** 2 + (xx - pos[1]) ** 2 <= radius * radius
    known[mask] = True
    known[true_map.start] = True
    known[true_map.goal] = True


def set_current_start(lunar: LunarMap, pos: Coord) -> LunarMap:
    return replace_lunar_start(lunar, pos)


def replace_lunar_start(lunar: LunarMap, pos: Coord) -> LunarMap:
    return LunarMap(
        lunar.size,
        lunar.elevation,
        lunar.slope,
        lunar.obstacle,
        lunar.crater_risk,
        lunar.regolith,
        lunar.illumination,
        lunar.communication,
        lunar.cost,
        pos,
        lunar.goal,
        lunar.battery_capacity,
    )


def first_step_from_plan(plan: Optional[List[Coord]], current: Coord) -> Optional[Coord]:
    if not plan or len(plan) < 2:
        return None
    if plan[0] != current:
        return None
    return plan[1]


def online_plan_step(method: str, belief: LunarMap, current: Coord, rng: np.random.Generator) -> Optional[Coord]:
    local = set_current_start(belief, current)
    if method == "Random-online":
        options = []
        for dy, dx in ACTIONS:
            nxt = (current[0] + dy, current[1] + dx)
            if 0 <= nxt[0] < belief.size and 0 <= nxt[1] < belief.size and not belief.obstacle[nxt]:
                options.append(nxt)
        if not options:
            return None
        options.sort(key=lambda p: heuristic(p, belief.goal))
        return options[int(rng.integers(0, min(len(options), 3)))]
    if method == "DFS-online":
        return first_step_from_plan(depth_first_search(local), current)
    if method == "BFS-online":
        return first_step_from_plan(breadth_first_search(local), current)
    if method == "Greedy-online":
        return first_step_from_plan(greedy_best_first(local), current)
    if method == "A* shortest-online":
        return first_step_from_plan(astar(local, risk_aware=False, battery_constrained=False), current)
    if method == "A* risk-aware-online":
        return first_step_from_plan(astar(local, risk_aware=True, battery_constrained=True), current)
    if method == "D* Lite-style":
        return first_step_from_plan(astar(local, risk_aware=True, battery_constrained=True), current)
    raise ValueError(method)


def run_online_method(
    true_map: LunarMap,
    scenario: Scenario,
    method: str,
    sensing_radius: int = 5,
    max_steps: int = 220,
    seed: int = 0,
) -> Tuple[List[Coord], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    known = np.zeros((true_map.size, true_map.size), dtype=bool)
    current = true_map.start
    energy_used = 0.0
    path = [current]
    replan_count = 0
    collision_fail = False
    start_time = time.perf_counter()

    for _ in range(max_steps):
        if current == true_map.goal:
            break
        sense(true_map, known, current, sensing_radius)
        belief = make_belief_map(true_map, known)
        nxt = online_plan_step(method, belief, current, rng)
        replan_count += 1
        if nxt is None:
            break
        if true_map.obstacle[nxt]:
            collision_fail = True
            break
        dy = nxt[0] - current[0]
        dx = nxt[1] - current[1]
        move = 2 ** 0.5 if dy and dx else 1.0
        energy_used += transition_energy(true_map, current, nxt, move)
        current = nxt
        path.append(current)
        if energy_used > scenario.battery_capacity:
            break

    elapsed = time.perf_counter() - start_time
    info = {
        "replan_count": replan_count,
        "collision_fail": int(collision_fail),
        "online_runtime_sec": elapsed,
        "known_cell_ratio": float(np.mean(known)),
    }
    return path, info


def run_online_method_trace(
    true_map: LunarMap,
    scenario: Scenario,
    method: str,
    sensing_radius: int = 5,
    max_steps: int = 220,
    seed: int = 0,
    snapshot_count: int = 6,
):
    rng = np.random.default_rng(seed)
    known = np.zeros((true_map.size, true_map.size), dtype=bool)
    current = true_map.start
    energy_used = 0.0
    path = [current]
    history = []

    for step in range(max_steps + 1):
        sense(true_map, known, current, sensing_radius)
        history.append(
            {
                "step": step,
                "known": known.copy(),
                "path": list(path),
                "pos": current,
                "energy_used": energy_used,
                "known_cell_ratio": float(np.mean(known)),
            }
        )
        if current == true_map.goal or energy_used > scenario.battery_capacity:
            break
        belief = make_belief_map(true_map, known)
        nxt = online_plan_step(method, belief, current, rng)
        if nxt is None or true_map.obstacle[nxt]:
            break
        dy = nxt[0] - current[0]
        dx = nxt[1] - current[1]
        move = 2 ** 0.5 if dy and dx else 1.0
        energy_used += transition_energy(true_map, current, nxt, move)
        current = nxt
        path.append(current)

    if len(history) <= snapshot_count:
        snapshots = history
    else:
        indices = np.unique(np.linspace(0, len(history) - 1, snapshot_count, dtype=int))
        snapshots = [history[int(idx)] for idx in indices]
    return path, snapshots


def plot_online_belief_sequence(
    true_map: LunarMap,
    scenario: Scenario,
    method: str,
    snapshots,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = snapshots[-1]["path"] if snapshots else []
    final_pos = final_path[-1] if final_path else true_map.start
    final_energy = snapshots[-1]["energy_used"] if snapshots else 0.0
    task_pass = final_pos == true_map.goal and final_energy <= true_map.battery_capacity
    final_status = "PASS" if task_pass else "FAIL"
    cols = 3
    rows = int(np.ceil(len(snapshots) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.2 * rows), constrained_layout=True)
    axes_flat = list(np.atleast_1d(axes).flat)
    terrain = base_rgb(true_map)
    for ax, snap in zip(axes_flat, snapshots):
        visible = terrain.copy()
        visible[~snap["known"]] = np.array([0.06, 0.06, 0.075])
        ax.imshow(visible)
        arr = np.array(snap["path"])
        if len(arr) > 1:
            ax.plot(arr[:, 1], arr[:, 0], color="#76ff03", lw=2.2)
        ax.scatter([true_map.start[1]], [true_map.start[0]], c="#00e676", s=70, marker="o", edgecolor="black", zorder=5)
        current_x = snap["pos"][1]
        current_y = snap["pos"][0]
        if snap["pos"] == true_map.goal:
            current_x += 0.55
        ax.scatter([current_x], [current_y], c="#fdd835", s=70, marker="s", edgecolor="black", zorder=7)
        ax.scatter([true_map.goal[1]], [true_map.goal[0]], c="white", s=155, marker="*", edgecolor="black", linewidths=1.2, zorder=8)
        ax.scatter([true_map.goal[1]], [true_map.goal[0]], c="#ff1744", s=105, marker="*", edgecolor="black", linewidths=1.0, zorder=9)
        battery_used = snap["energy_used"] / true_map.battery_capacity
        battery_label = "100%+" if battery_used > 1.0 else f"{battery_used:.0%}"
        status = "PASS" if snap["pos"] == true_map.goal and snap["energy_used"] <= true_map.battery_capacity else "RUN/FAIL"
        ax.set_title(
            f"step {snap['step']} | known {snap['known_cell_ratio']:.0%} | battery {battery_label} | {status}",
            fontsize=10,
            weight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_flat[len(snapshots):]:
        ax.axis("off")
    fig.savefig(out_dir / f"{scenario.name}_{method.lower().replace('*', 'astar').replace(' ', '_')}_belief_sequence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_online_experiment(out_dir: Path, model_dir: Path, sensing_radius: int = 5) -> pd.DataFrame:
    rows = []
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    online_methods = [
        "Random-online",
        "DFS-online",
        "BFS-online",
        "Greedy-online",
        "A* shortest-online",
        "A* risk-aware-online",
        "D* Lite-style",
    ]

    for scenario in default_scenarios():
        true_map = generate_lunar_map(scenario)
        paths = {}
        for method in online_methods:
            path, info = run_online_method(true_map, scenario, method, sensing_radius=sensing_radius, seed=scenario.seed + 77)
            paths[method.replace("-online", "")] = path
            row = path_metrics(true_map, path, method, scenario)
            row.update(info)
            row["sensing_radius"] = sensing_radius
            rows.append(row)

        for method_name, model_cls in [("DQN-online", DQN), ("PPO-online", PPO)]:
            model_name = "dqn" if method_name.startswith("DQN") else "ppo"
            model = model_cls.load(model_dir / f"{scenario.name}_{model_name}.zip", device="cuda")
            path = execute_path_with_battery(true_map, deep_rl_policy_path(true_map, model))
            paths[method_name.replace("-online", "")] = path
            row = path_metrics(true_map, path, method_name, scenario)
            row.update({"replan_count": 0, "collision_fail": 0, "online_runtime_sec": 0.0, "known_cell_ratio": np.nan, "sensing_radius": sensing_radius})
            rows.append(row)

        plot_path_panels(true_map, scenario, paths, fig_dir / "online")
        if scenario.name in {"shadow_comm", "low_battery_bad_case"}:
            _, snapshots = run_online_method_trace(
                true_map,
                scenario,
                "D* Lite-style",
                sensing_radius=sensing_radius,
                seed=scenario.seed + 77,
            )
            plot_online_belief_sequence(true_map, scenario, "D* Lite-style", snapshots, fig_dir / "online")

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "online_metrics.csv", index=False)
    plot_online_outcome(results, fig_dir)
    return results


def plot_online_outcome(results: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    outcome = results.pivot(index="method", columns="scenario", values="task_success")
    labels = outcome.map(lambda value: "PASS" if value == 1 else "FAIL")
    fig, ax = plt.subplots(figsize=(11, 5.2))
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
    fig.savefig(fig_dir / "online_task_outcome_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def online_report_section(out_dir: Path) -> str:
    path = out_dir / "online_metrics.csv"
    if not path.exists():
        return "_Online local-view experiment has not been generated yet._\n"
    df = pd.read_csv(path)
    summary = df.groupby("method")["task_success"].agg(["sum", "count"]).reset_index()
    summary["pass_rate"] = summary["sum"] / summary["count"]
    return "\n".join(
        [
            "![Online task outcome matrix](results/figures/online_task_outcome_matrix.png)",
            "",
            "### Online Result Summary",
            "",
            df_to_markdown(summary.round(4)),
            "",
            "### Online Raw Metrics",
            "",
            df_to_markdown(df[["scenario", "method", "path_found", "energy_feasible", "task_success", "energy", "battery_capacity", "replan_count", "collision_fail", "known_cell_ratio"]].round(4)),
            "",
        ]
    )
