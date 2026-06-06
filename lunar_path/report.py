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


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _pass_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("method")["task_success"]
        .agg(["sum", "count"])
        .reset_index()
        .assign(pass_rate=lambda df: df["sum"] / df["count"])
        .sort_values(["pass_rate", "method"], ascending=[True, True])
    )


def _sort_pivot_by_overall(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    value_cols = [c for c in df.columns if c != "method"]
    table = df.copy()
    table["_overall"] = table[value_cols].mean(axis=1)
    table = table.sort_values(["_overall", "method"], ascending=[True, True]).drop(columns=["_overall"])
    return table


def write_report(metrics: pd.DataFrame, scenarios: List[Scenario], out_path: Path) -> None:
    results_dir = out_path.parent / "results"
    generalization = _read_optional(results_dir / "generalization_metrics.csv")
    online = _read_optional(results_dir / "online_metrics.csv")
    scale_up = _read_optional(results_dir / "scale_up_summary.csv")
    scale_up_metrics = _read_optional(results_dir / "scale_up_metrics.csv")

    success = metrics[metrics["path_found"] == 1].copy()
    best_cost = success.sort_values("total_cost").groupby("scenario").first().reset_index()
    best_energy = success.sort_values("energy").groupby("scenario").first().reset_index()

    scenario_table = pd.DataFrame(
        [
            {
                "scenario": s.name,
                "terrain": s.title,
                "craters": s.crater_count,
                "rock_density": s.rock_density,
                "shadow_patches": s.shadow_patches,
                "comm_stations": len(s.comm_stations),
                "battery_capacity": s.battery_capacity,
            }
            for s in scenarios
        ]
    )

    lines = [
        "# Energy-Aware Path Planning for Lunar Industrial Rovers under Terrain Risk and Local Sensing Constraints",
        "",
        "## Abstract",
        "",
        "This report studies energy-constrained path planning for a lunar industrial rover operating on moon-like terrain. The simulated environment contains DEM-like elevation, crater and rock obstacles, slope risk, soft regolith, illumination variation, communication coverage, and scenario-specific battery capacity. The study first evaluates offline full-map planning, where the complete terrain map is known before planning. It then adds two extensions: reinforcement-learning transfer to unseen random maps and online local-view replanning with incremental map updates. Results show that distance-only planning is insufficient under lunar terrain constraints. A risk-aware A* planner, which explicitly optimizes energy and terrain risk, is the most reliable method in both case-level and scale-up experiments.",
        "",
        "**Keywords:** lunar rover, path planning, energy constraint, risk-aware A*, D* Lite-style replanning, reinforcement learning, local sensing",
        "",
        "## 1. Introduction",
        "",
        "Future lunar industrial activity will require mobile robots to transport equipment, resources, and samples across unstructured terrain. Unlike terrestrial road navigation, lunar surface mobility is affected by craters, rocks, steep slopes, soft regolith, poor illumination, communication gaps, and strict battery limits. A shortest geometric path may therefore fail even when it is collision-free.",
        "",
        "The goal of this project is to compare classical search, heuristic planning, risk-aware planning, and reinforcement learning under these lunar constraints. The central question is whether a planner that explicitly models terrain-dependent energy and mission risk can outperform simpler path-planning baselines.",
        "",
        "## 2. Problem Formulation",
        "",
        "The rover moves on a `45 x 45` grid map. A task starts at a lunar base location and ends at a target industrial site. A route is successful only if the rover reaches the goal and its accumulated energy consumption does not exceed the scenario battery capacity.",
        "",
        "The main experiment assumes a **fully known offline lunar terrain map**. Before planning starts, the rover has access to all terrain layers. This is not a SLAM experiment. Online perception and partial map knowledge are studied separately in Section 8.",
        "",
        "Each cell stores the following map layers:",
        "",
        "| Layer | Meaning | Role in planning |",
        "|---|---|---|",
        "| elevation | DEM-like terrain height | Used for uphill energy cost |",
        "| slope | local slope | Increases energy and terrain risk |",
        "| obstacle | rock or crater core | Impassable cell |",
        "| crater_risk | crater rim hazard | Adds risk cost |",
        "| regolith | soft soil level | Increases traction energy cost |",
        "| illumination | sunlight level | Low illumination increases energy cost |",
        "| communication | communication quality | Poor coverage increases mission risk |",
        "",
        "The transition-level energy model is:",
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
        "The total risk-aware transition cost is:",
        "",
        "```text",
        "transition_cost = transition_energy",
        "                + 2.2 * crater_risk(s')",
        "                + 1.0 * slope(s')",
        "                + 1.3 * (1 - communication(s'))",
        "```",
        "",
        "Battery use is accumulated during execution. If cumulative energy exceeds `battery_capacity`, the route immediately fails.",
        "",
        "## 3. Methods",
        "",
        "Eight methods are compared in the offline full-map experiment:",
        "",
        "| Method | Description | Main limitation |",
        "|---|---|---|",
        "| Random | Random feasible movement baseline | No global objective |",
        "| DFS | Depth-first graph search | Can produce long inefficient paths |",
        "| BFS | Breadth-first graph search | Optimizes step count, not energy |",
        "| Greedy | Local best-first movement | Can be locally trapped or energy-myopic |",
        "| A* shortest | A* with geometric path cost | Ignores terrain energy and risk |",
        "| A* risk-aware | A* with energy-risk transition cost and battery pruning | Requires known cost map |",
        "| DQN | Deep Q-network trained per scenario | Limited map awareness in low-dimensional observation |",
        "| PPO | Policy-gradient RL trained per scenario | Same limitation as DQN |",
        "",
        "DQN and PPO use a lightweight safety executor during rollout. Invalid actions, repeated cells, and immediately battery-depleting actions are rejected. This enforces rover safety constraints but does not give the RL agent a global planner.",
        "",
        "## 4. Experimental Setup",
        "",
        "Six lunar scenario families are generated. Battery capacities are selected from pilot runs to avoid trivial all-pass or all-fail outcomes.",
        "",
        df_to_markdown(scenario_table.round(4)),
        "",
        "The report evaluates four experimental settings:",
        "",
        "1. **Offline full-map case study:** all eight methods are tested on the six representative scenario maps.",
        "2. **RL unseen-map test:** saved DQN/PPO models are tested on same-family maps generated with new random seeds.",
        "3. **Online local-view replanning:** the rover incrementally reveals a local sensing window and replans as the map is updated.",
        "4. **Scale-up test:** multiple random maps per scenario family are generated to estimate pass rates statistically.",
        "",
        "## 5. Offline Full-Map Results",
        "",
        "**Figure 1. Pass/fail outcome matrix for the offline full-map experiment.**",
        "",
        "![Task outcome matrix](results/figures/task_outcome_matrix.png)",
        "",
        "**Figure 2. Compact cross-method metric matrix. Values are normalized within each scenario; lower is better.**",
        "",
        "![Metrics comparison](results/figures/metrics_comparison.png)",
        "",
        "**Figure 3. Normalized metric heatmap. Lower values are better.**",
        "",
        "![Metrics heatmap](results/figures/metrics_heatmap.png)",
        "",
        "**Figure 4. Battery energy margin by method and scenario.**",
        "",
        "![Battery energy margin](results/figures/battery_energy_margin.png)",
        "",
        "### 5.1 Offline Summary",
        "",
        df_to_markdown(_pass_summary(metrics).round(4)),
        "",
        "### 5.2 Lowest-Cost and Lowest-Energy Routes",
        "",
        "**Lowest total cost by scenario:**",
        "",
        df_to_markdown(best_cost[["scenario", "method", "total_cost", "path_length", "energy", "terrain_risk"]].round(4)),
        "",
        "**Lowest energy by scenario:**",
        "",
        df_to_markdown(best_energy[["scenario", "method", "energy", "total_cost", "path_length", "terrain_risk"]].round(4)),
        "",
        "### 5.3 Representative Path Visualizations",
        "",
        "The following figures show representative combined and per-method path visualizations. Overlapping paths in combined figures are slightly offset for readability.",
        "",
    ]

    for scenario in scenarios:
        lines.extend(
            [
                f"**Figure. Combined paths for `{scenario.name}`.**",
                "",
                f"![{scenario.title} paths](results/figures/{scenario.name}_paths.png)",
                "",
                f"**Figure. Individual method paths for `{scenario.name}`. Failed trajectories are marked with an `X`.**",
                "",
                f"![{scenario.title} individual method paths](results/figures/{scenario.name}_path_panels.png)",
                "",
            ]
        )

    lines.extend(
        [
            "## 6. Reinforcement Learning Training and Unseen-Map Test",
            "",
            "DQN and PPO are trained on their corresponding scenario maps. To test whether these policies generalize beyond the training map, the saved models are evaluated on unseen maps generated from the same scenario settings but different random seeds. No additional training is performed for this test.",
            "",
            "**Figure 5. DQN/PPO training reward curves.**",
            "",
            "![RL training reward](results/figures/rl_training_reward.png)",
            "",
            "**Figure 6. DQN/PPO unseen-map pass/fail matrix.**",
            "",
            "![RL generalization matrix](results/figures/rl_generalization_matrix.png)",
            "",
        ]
    )

    if not generalization.empty:
        gen_summary = generalization.groupby("method")["task_success"].agg(["sum", "count"]).reset_index()
        gen_summary["pass_rate"] = gen_summary["sum"] / gen_summary["count"]
        gen_summary = gen_summary.sort_values(["pass_rate", "method"], ascending=[True, True])
        lines.extend(
            [
                "### 6.1 Unseen-Map Summary",
                "",
                df_to_markdown(gen_summary.round(4)),
                "",
                "The unseen-map test shows whether the learned policies transfer to new maps. In this implementation, RL policies have limited map-structural input, so their transfer performance should be interpreted cautiously.",
                "",
            ]
        )

    lines.extend(
        [
            "## 7. Online Local-View Replanning",
            "",
            "The offline assumption is relaxed in this experiment. The rover starts with an unknown map and reveals only a circular local sensing window around its current position. Unknown cells are treated as traversable with neutral terrain estimates until observed. Methods replan using the currently known map. A D* Lite-style baseline is implemented as repeated risk-aware replanning after each sensing update; it captures the behavioral idea of D*/D* Lite without implementing incremental priority-queue optimization.",
            "",
            "**Figure 7. Online local-view pass/fail matrix.**",
            "",
            "![Online task outcome matrix](results/figures/online_task_outcome_matrix.png)",
            "",
            "### 7.1 Local Belief Update Examples",
            "",
            "Dark cells are unknown. Revealed cells are inside accumulated local sensing windows. The yellow square marks the current rover position. Battery use is shown as a percentage of the scenario-specific budget.",
            "",
            "**Figure 8. Failure example: D* Lite-style replanning in `shadow_comm`. The rover nearly reaches the target but exceeds the battery budget.**",
            "",
            "![Shadow communication online belief update](results/figures/online/shadow_comm_dastar_lite-style_belief_sequence.png)",
            "",
            "**Figure 9. Success example: D* Lite-style replanning in `low_battery_bad_case`. The rover reaches the target within the battery budget.**",
            "",
            "![Low battery online belief update](results/figures/online/low_battery_bad_case_dastar_lite-style_belief_sequence.png)",
            "",
        ]
    )

    if not online.empty:
        online_summary = online.groupby("method")["task_success"].agg(["sum", "count"]).reset_index()
        online_summary["pass_rate"] = online_summary["sum"] / online_summary["count"]
        online_summary = online_summary.sort_values(["pass_rate", "method"], ascending=[True, True])
        lines.extend(
            [
                "### 7.2 Online Summary",
                "",
                df_to_markdown(online_summary.round(4)),
                "",
            ]
        )

    lines.extend(
        [
            "## 8. Scale-Up Experiment",
            "",
            "The case-level experiments are extended by generating multiple random maps for each scenario family. This provides a more robust estimate of performance than a small number of hand-picked maps.",
            "",
        ]
    )

    if not scale_up.empty:
        overall = (
            scale_up.groupby(["setting", "method"])
            .agg(
                pass_rate=("pass_rate", "mean"),
                mean_energy=("mean_energy", "mean"),
                mean_path_length=("mean_path_length", "mean"),
            )
            .reset_index()
            .sort_values(["setting", "pass_rate", "method"], ascending=[True, True, True])
        )
        if not scale_up_metrics.empty:
            n_scenarios = scale_up_metrics["scenario"].nunique()
            n_seeds = int(scale_up_metrics.groupby(["setting", "scenario"])["sample_seed"].nunique().max())
            n_map_setting_pairs = scale_up_metrics[["setting", "scenario", "sample_seed"]].drop_duplicates().shape[0]
            n_method_evals = len(scale_up_metrics)
            lines.extend(
                [
                    f"This scale-up experiment uses `{n_scenarios}` scenario families and `{n_seeds}` random maps per scenario family for each setting. Across the offline and online settings, this corresponds to `{n_map_setting_pairs}` generated map-setting pairs and `{n_method_evals}` method evaluations.",
                    "",
                ]
            )
        lines.extend(
            [
                "**Figure 10. Scale-up pass rates for the offline full-map setting.**",
                "",
                "![Offline scale-up pass rate](results/figures/scale_up_offline_full_map_pass_rate.png)",
                "",
                "**Figure 11. Scale-up pass rates for the online local-view setting.**",
                "",
                "![Online scale-up pass rate](results/figures/scale_up_online_local_view_pass_rate.png)",
                "",
                "**Figure 12. Overall scale-up pass rate by method.**",
                "",
                "![Overall scale-up pass rate](results/figures/scale_up_overall_pass_rate.png)",
                "",
                "### 8.1 Scale-Up Aggregate Metrics",
                "",
                df_to_markdown(overall.round(4)),
                "",
            ]
        )
        offline_table = _sort_pivot_by_overall(scale_up[scale_up["setting"] == "offline_full_map"].pivot(index="method", columns="scenario", values="pass_rate").reset_index())
        online_table = _sort_pivot_by_overall(scale_up[scale_up["setting"] == "online_local_view"].pivot(index="method", columns="scenario", values="pass_rate").reset_index())
        lines.extend(
            [
                "### 8.2 Offline Pass Rate by Scenario Family",
                "",
                df_to_markdown(offline_table.round(4)),
                "",
                "### 8.3 Online Pass Rate by Scenario Family",
                "",
                df_to_markdown(online_table.round(4)),
                "",
            ]
        )

    lines.extend(
        [
            "## 9. Discussion",
            "",
            "The experiments support three main observations. First, shortest-path planning is not sufficient for lunar terrain because energy consumption depends strongly on slope, regolith, uphill movement, and illumination. Second, explicit cost modeling is valuable: A* risk-aware performs strongly because its planning objective matches the evaluation objective. Third, online local-view planning is harder than offline planning because early choices are made with incomplete terrain knowledge, which can lead to energy-inefficient routes before hazards are fully revealed.",
            "",
            "The reinforcement-learning results are mixed. DQN and PPO can find feasible routes in several settings, but their low-dimensional observation does not include a local map patch. As a result, the learned policies do not consistently outperform model-based risk-aware planning. A stronger RL formulation would likely require multi-channel local map observations and a CNN-based policy.",
            "",
            "## 10. Limitations",
            "",
            "- The lunar terrain is a simplified grid simulation rather than a high-fidelity rover dynamics simulator.",
            "- The offline experiment assumes a fully known map.",
            "- The online experiment uses a simplified local sensing model and neutral assumptions for unknown cells.",
            "- D* Lite-style replanning captures repeated replanning behavior but not optimized incremental D* Lite data structures.",
            "- DQN/PPO use compact state features rather than image-like local terrain patches.",
            "",
            "## 11. Conclusion",
            "",
            "This study shows that lunar industrial rover navigation should be treated as an energy-constrained and risk-aware planning problem rather than a shortest-path problem. Across representative cases and random-seed scale-up tests, risk-aware A* is the most reliable method because it directly optimizes terrain-dependent energy and mission risk while respecting battery capacity. Online local-view replanning reduces performance relative to full-map planning, but D* Lite-style and risk-aware replanning remain competitive when the map is incrementally revealed.",
            "",
            "## Output Files",
            "",
            "- Main report: `experiments/report.md`",
            "- Main metrics: `experiments/results/metrics.csv`",
            "- Online metrics: `experiments/results/online_metrics.csv`",
            "- Scale-up metrics: `experiments/results/scale_up_metrics.csv`",
            "- Scale-up summary: `experiments/results/scale_up_summary.csv`",
            "- RL model weights: `experiments/results/models/`",
            "- Figures: `experiments/results/figures/`",
            "",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
