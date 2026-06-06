from pathlib import Path
from typing import List

import pandas as pd

from lunar_path.environment import Scenario


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    table = df.copy()
    headers = [str(col) for col in table.columns]
    rows = [[str(value) for value in row] for row in table.to_numpy()]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    header_line = "| " + " | ".join(header.ljust(width) for header, width in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * width for width in widths) + " |"
    row_lines = ["| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + row_lines)


def write_report(metrics: pd.DataFrame, scenarios: List[Scenario], out_path: Path) -> None:
    success = metrics[metrics["path_found"] == 1].copy()
    best_cost = success.sort_values("total_cost").groupby("scenario").first().reset_index()
    best_energy = success.sort_values("energy").groupby("scenario").first().reset_index()
    generalization_path = out_path.parent / "results" / "generalization_metrics.csv"
    generalization = pd.read_csv(generalization_path) if generalization_path.exists() else pd.DataFrame()
    online_path = out_path.parent / "results" / "online_metrics.csv"
    online = pd.read_csv(online_path) if online_path.exists() else pd.DataFrame()

    lines = [
        "# Lunar Industrial Path Planning Experiment Report",
        "",
        "## 1. Objective",
        "",
        "This project studies path planning for an autonomous transport rover in a lunar industrial base. The simulated terrain is designed to resemble a simplified lunar surface, including crater fields, rock obstacles, elevation variation, steep ridges, soft regolith, permanent shadow regions, illumination differences, and communication blind zones.",
        "",
        "The experiment compares eight methods:",
        "",
        "- **Random**: a stochastic baseline that samples feasible moves.",
        "- **DFS**: depth-first search, a blind graph-search baseline that can produce long and energy-inefficient routes.",
        "- **BFS**: breadth-first search, an unweighted graph-search baseline that minimizes the number of grid steps but ignores terrain energy.",
        "- **Greedy**: a local best-first baseline using distance-to-goal and terrain cost.",
        "- **A\\* shortest**: a geometric shortest-path baseline using distance and impassable obstacles.",
        "- **A\\* risk-aware**: a cost-aware A\\* planner using slope, crater risk, regolith, illumination, and communication quality.",
        "- **DQN**: a deep Q-network reinforcement learning agent trained with Stable-Baselines3.",
        "- **PPO**: a policy-gradient reinforcement learning agent trained with Stable-Baselines3.",
        "",
        "DQN and PPO are evaluated with a lightweight safety executor: invalid moves, repeated cells, and immediately battery-depleting moves are rejected during rollout. This does not provide a global path solution to the RL agents; it only enforces rover safety constraints during execution.",
        "",
        "## 2. Inputs and Outputs",
        "",
        "### Inputs",
        "",
        "- Map size: `45 x 45` grid cells.",
        "- Start: lunar base core area.",
        "- Goal: resource extraction site or industrial target facility.",
        "- Terrain layers: elevation, slope, obstacles, crater risk, soft regolith, illumination, and communication coverage.",
        "- Action space: 8 movement directions, including cardinal and diagonal moves.",
        "- RL reward: terminal reward for reaching the goal, terrain-dependent transition cost, invalid-move penalty, and potential-based distance shaping.",
        "- Fairness control: all methods use the same map, start, goal, action directions, obstacle constraints, and evaluation metrics. The RL base reward uses the same transition cost optimized by risk-aware A\\*; distance shaping is potential-based and is used only to accelerate training.",
        "- Battery constraint: a route is counted as a full task success only if a path is found and its estimated energy consumption does not exceed the scenario battery capacity.",
        "",
        "### Outputs",
        "",
        "- Lunar environment visualization for each scenario.",
        "- Path comparison visualization for each scenario.",
        "- Deep RL training reward curves.",
        "- Completed training episode chart.",
        "- Cross-method metric comparison chart.",
        "- Normalized metric heatmap.",
        "- Battery feasibility and energy margin charts.",
        "- Saved DQN/PPO model weights for each scenario.",
        "- Raw metric CSV file.",
        "",
        "## 3. Lunar Simulation Environment",
        "",
        "### 3.1 Mapping Assumption",
        "",
        "This experiment assumes a **fully known offline lunar terrain map**. Before planning starts, the rover is assumed to have access to the complete simulated DEM-derived environment, including elevation, slope, obstacle locations, crater risk, regolith, illumination, communication coverage, and battery capacity.",
        "",
        "This means the current work is an offline path-planning study, not an online SLAM or exploration problem. Online perception, localization uncertainty, incremental mapping, dynamic obstacle discovery, and real-time map updates are outside the scope of this experiment.",
        "",
        "The planning problem can therefore be stated as:",
        "",
        "> Given a known lunar terrain map with DEM-derived slope, terrain risk, illumination, communication coverage, and battery constraints, plan an energy-feasible and risk-aware path for a lunar industrial rover.",
        "",
        "### 3.2 Terrain Layers",
        "",
        "| Variable | Meaning | Planning effect |",
        "|---|---|---|",
        "| elevation | Terrain height | Creates lunar ridges and depressions |",
        "| slope | Local slope | Higher slope increases energy use and risk; extreme slope is impassable |",
        "| obstacle | Rock or crater core | Impassable cell |",
        "| crater_risk | Crater rim risk | Adds terrain hazard cost |",
        "| regolith | Soft regolith level | Adds energy cost and traction risk |",
        "| illumination | Sunlight intensity | Low illumination increases energy and thermal risk |",
        "| communication | Communication quality | Poor coverage increases mission risk |",
        "",
        "The map also stores a cell-level terrain score for visualization and local greedy scoring:",
        "",
        "```text",
        "Cost = 1",
        "     + 2.8 * slope",
        "     + 2.2 * crater_risk",
        "     + 1.7 * regolith",
        "     + 1.6 * (1 - illumination)",
        "     + 1.3 * (1 - communication)",
        "```",
        "",
        "Energy is explicitly modeled at the transition level. Moving across different terrain consumes different energy. For a movement from cell `s` to next cell `s'`, the energy model is:",
        "",
        "```text",
        "transition_energy = move_distance",
        "                  * (1",
        "                     + 2.2 * slope(s')",
        "                     + 1.7 * regolith(s')",
        "                     + 1.1 * uphill_elevation_gain",
        "                     + 0.8 * (1 - illumination(s')))",
        "```",
        "",
        "The total mission transition cost used by risk-aware A\\* and the RL base reward is:",
        "",
        "```text",
        "transition_cost = transition_energy",
        "                + 2.2 * crater_risk(s')",
        "                + 1.0 * slope(s')",
        "                + 1.3 * (1 - communication(s'))",
        "RL_base_reward = -transition_cost",
        "```",
        "",
        "During execution, energy is accumulated after every movement. If cumulative energy exceeds `battery_capacity`, the route immediately fails and the executed trajectory stops at the depletion point. This applies to Random, Greedy, A\\*, DQN, and PPO rollouts.",
        "",
        "## 4. Experimental Scenarios",
        "",
    ]

    for scenario in scenarios:
        lines.extend([
            f"### {scenario.name}: {scenario.title}",
            "",
            f"- Crater count: {scenario.crater_count}",
            f"- Rock density: {scenario.rock_density}",
            f"- Soft regolith patches: {scenario.regolith_patches}",
            f"- Shadow patches: {scenario.shadow_patches}",
            f"- Communication stations: {len(scenario.comm_stations)}",
            f"- Elevation variation scale: {scenario.slope_scale}",
            f"- RL training timesteps per deep RL method: {scenario.rl_timesteps}",
            f"- Battery capacity: {scenario.battery_capacity}",
            "",
            f"![{scenario.title} environment](results/figures/{scenario.name}_environment.png)",
            "",
            f"![{scenario.title} paths](results/figures/{scenario.name}_paths.png)",
            "",
            f"![{scenario.title} individual method paths](results/figures/{scenario.name}_path_panels.png)",
            "",
        ])

    lines.extend([
        "## 5. Visual Results",
        "",
        "### 5.1 Metric Comparison",
        "",
        "![Metrics comparison](results/figures/metrics_comparison.png)",
        "",
        "### 5.2 Normalized Metric Heatmap",
        "",
        "Lower normalized values indicate better performance for the corresponding metric.",
        "",
        "![Metrics heatmap](results/figures/metrics_heatmap.png)",
        "",
        "### 5.3 Battery Constraint Results",
        "",
        "![Battery task success](results/figures/battery_task_success.png)",
        "",
        "![Battery energy margin](results/figures/battery_energy_margin.png)",
        "",
        "![Task outcome matrix](results/figures/task_outcome_matrix.png)",
        "",
        "### 5.4 Deep RL Training Curves",
        "",
        "![RL training reward](results/figures/rl_training_reward.png)",
        "",
        "![RL training episodes](results/figures/rl_training_episodes.png)",
        "",
        "## 6. Result Summary",
        "",
        "### 6.1 Raw Metric Table",
        "",
        df_to_markdown(metrics.round(4)),
        "",
        "### 6.2 Lowest Total Cost Method per Scenario",
        "",
        df_to_markdown(best_cost[["scenario", "method", "total_cost", "path_length", "energy", "terrain_risk"]].round(4)),
        "",
        "### 6.3 Lowest Energy Method per Scenario",
        "",
        df_to_markdown(best_energy[["scenario", "method", "energy", "total_cost", "path_length", "terrain_risk"]].round(4)),
        "",
        "## 7. Analysis",
        "",
        "1. **Random** is included only as a weak lower-bound baseline. It usually fails in dense crater or high-risk maps because it has no global objective.",
        "",
        "2. **Greedy** improves over Random by moving toward the goal, but it is local and can get trapped near crater rims or obstacle pockets.",
        "",
        "3. **DFS** and **BFS** are useful classical baselines. BFS can find a step-short path, while DFS may find a much longer route depending on search order. Neither method understands energy, illumination, communication, or crater risk during planning.",
        "",
        "4. **A\\* shortest** usually finds a short geometric route, but it may cross crater rims, steep slopes, shadow regions, or low-communication zones because it ignores mission risk.",
        "",
        "5. **A\\* risk-aware** explicitly models lunar terrain cost. It often selects a longer route, but the route is safer and more realistic for lunar industrial logistics.",
        "",
        "6. **DQN** and **PPO** learn policies through environment interaction. They can produce feasible paths when reward shaping is sufficient, but their stability depends on training budget and scenario complexity. In the low-battery bad case, the neural policies avoid immediate battery depletion but still fail to reach the goal, which is an important negative result.",
        "",
        "7. The battery capacities are intentionally tuned to create mixed outcomes rather than an all-pass or all-fail benchmark. In easier settings, several classical and RL methods can finish. In harder settings, methods fail for different reasons: blind search wastes energy, shortest-path search ignores terrain cost, greedy planning can be locally efficient but not globally safe, and RL policies may conserve energy without reaching the goal. A\\* risk-aware is the only method designed to explicitly optimize the same energy-risk cost used by the evaluation, which explains its stronger pass rate.",
        "",
        "8. Lunar path planning is not a normal shortest-path problem. After slope, illumination, communication, regolith, and battery capacity are introduced, the shortest path and the best mission path often differ.",
        "",
        "## 8. Conclusion",
        "",
        "The experiment demonstrates that moon-like terrain constraints change the path-planning objective. Distance-only planning is not enough for lunar industrial transportation. A risk-aware planner is a strong baseline for known static maps, while deep reinforcement learning becomes more relevant when future tasks include unknown terrain, dynamic hazards, multi-rover coordination, or long-horizon task scheduling.",
        "",
        "## 9. Extra Experiment: RL Generalization to Unseen Maps",
        "",
        "The main RL experiments train and evaluate DQN/PPO on the same scenario map. To test whether the learned policies generalize, an extra experiment evaluates the saved DQN/PPO models on unseen maps generated from the same scenario settings but with different random seeds. No additional training is performed.",
        "",
    ])

    if generalization.empty:
        lines.extend([
            "_Generalization results have not been generated yet._",
            "",
        ])
    else:
        gen_summary = generalization.groupby("method")["task_success"].agg(["sum", "count"]).reset_index()
        gen_summary["pass_rate"] = gen_summary["sum"] / gen_summary["count"]
        lines.extend([
            "### 9.1 Generalization Result Table",
            "",
            "![RL generalization matrix](results/figures/rl_generalization_matrix.png)",
            "",
            df_to_markdown(generalization[["train_scenario", "method", "train_seed", "test_seed", "path_found", "energy_feasible", "task_success", "energy", "battery_capacity", "energy_margin"]].round(4)),
            "",
            "### 9.2 Generalization Summary",
            "",
            df_to_markdown(gen_summary.round(4)),
            "",
            "The result measures scenario-specific policy transfer to a new map seed. If pass rates are low, the conclusion is that the current DQN/PPO policies are not robust map-generalization planners; they mainly learn behavior for the map distribution seen during training.",
            "",
        ])

    lines.extend([
        "## 10. Extra Experiment: Online Local-View Replanning",
        "",
        "The main experiment assumes a fully known offline map. A second extra experiment relaxes this assumption: the rover starts with an unknown map and updates only a local circular sensing window around its current position. Unknown cells are treated as traversable with neutral terrain estimates until they are observed.",
        "",
        "This online experiment evaluates repeated local replanning methods, including Random-online, DFS-online, BFS-online, Greedy-online, A\\* shortest-online, A\\* risk-aware-online, and a **D\\* Lite-style** replanning baseline. The D\\* Lite-style method is implemented as local-map risk-aware replanning after each sensing update. It captures the experiment-level behavior of D\\*/D\\* Lite, namely replanning as the map is incrementally revealed, but it is not optimized for D\\* Lite's incremental priority-queue efficiency.",
        "",
        "DQN/PPO are also evaluated in this section using their saved policies and safety executor, but they are not retrained specifically for partial observability.",
        "",
    ])

    if online.empty:
        lines.extend([
            "_Online local-view results have not been generated yet._",
            "",
        ])
    else:
        online_summary = online.groupby("method")["task_success"].agg(["sum", "count"]).reset_index()
        online_summary["pass_rate"] = online_summary["sum"] / online_summary["count"]
        lines.extend([
            "![Online task outcome matrix](results/figures/online_task_outcome_matrix.png)",
            "",
            "### 10.1 Local Belief Update Visualization",
            "",
            "In the following figures, dark cells are still unknown to the rover. Revealed cells are inside the local sensing windows accumulated along the executed trajectory. The yellow square marks the current rover position at each snapshot.",
            "",
            "![Shadow communication online belief update](results/figures/online/shadow_comm_dastar_lite-style_belief_sequence.png)",
            "",
            "![Low battery online belief update](results/figures/online/low_battery_bad_case_dastar_lite-style_belief_sequence.png)",
            "",
            "### 10.2 Online Summary",
            "",
            df_to_markdown(online_summary.round(4)),
            "",
            "### 10.3 Online Raw Metrics",
            "",
            df_to_markdown(online[["scenario", "method", "path_found", "energy_feasible", "task_success", "energy", "battery_capacity", "replan_count", "collision_fail", "known_cell_ratio"]].round(4)),
            "",
            "The online results should be interpreted differently from the offline results. Online methods may fail because the local view does not reveal enough terrain structure early enough, because they choose an energy-inefficient route before discovering later hazards, or because their replanning strategy is too myopic under battery constraints.",
            "",
        ])

    lines.extend([
        "## 11. Output Files",
        "",
        "- Main runner: `scripts/run_lunar_path_experiments.py`",
        "- Environment module: `lunar_path/environment.py`",
        "- Method modules: `lunar_path/methods/`",
        "- Metric CSV: `experiments/results/metrics.csv`",
        "- RL generalization CSV: `experiments/results/generalization_metrics.csv`",
        "- Online local-view CSV: `experiments/results/online_metrics.csv`",
        "- RL training logs: `experiments/results/rl_training_logs.csv`",
        "- Saved RL weights: `experiments/results/models/`",
        "- Figure directory: `experiments/results/figures/`",
        "- Report file: `experiments/report.md`",
        "",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
