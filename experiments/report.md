# Energy-Aware Path Planning for Lunar Industrial Rovers under Terrain Risk and Local Sensing Constraints

## Abstract

This report studies energy-constrained path planning for a lunar industrial rover operating on moon-like terrain. The simulated environment contains DEM-like elevation, crater and rock obstacles, slope risk, soft regolith, illumination variation, communication coverage, and scenario-specific battery capacity. The study first evaluates offline full-map planning, where the complete terrain map is known before planning. It then adds two extensions: reinforcement-learning transfer to unseen random maps and online local-view replanning with incremental map updates. Results show that distance-only planning is insufficient under lunar terrain constraints. A risk-aware A* planner, which explicitly optimizes energy and terrain risk, is the most reliable method in both case-level and scale-up experiments.

**Keywords:** lunar rover, path planning, energy constraint, risk-aware A*, D* Lite-style replanning, reinforcement learning, local sensing

## 1. Introduction

Future lunar industrial activity will require mobile robots to transport equipment, resources, and samples across unstructured terrain. Unlike terrestrial road navigation, lunar surface mobility is affected by craters, rocks, steep slopes, soft regolith, poor illumination, communication gaps, and strict battery limits. A shortest geometric path may therefore fail even when it is collision-free.

The goal of this project is to compare classical search, heuristic planning, risk-aware planning, and reinforcement learning under these lunar constraints. The central question is whether a planner that explicitly models terrain-dependent energy and mission risk can outperform simpler path-planning baselines.

## 2. Problem Formulation

The rover moves on a `45 x 45` grid map. A task starts at a lunar base location and ends at a target industrial site. A route is successful only if the rover reaches the goal and its accumulated energy consumption does not exceed the scenario battery capacity.

The main experiment assumes a **fully known offline lunar terrain map**. Before planning starts, the rover has access to all terrain layers. This is not a SLAM experiment. Online perception and partial map knowledge are studied separately in Section 8.

Each cell stores the following map layers:

| Layer | Meaning | Role in planning |
|---|---|---|
| elevation | DEM-like terrain height | Used for uphill energy cost |
| slope | local slope | Increases energy and terrain risk |
| obstacle | rock or crater core | Impassable cell |
| crater_risk | crater rim hazard | Adds risk cost |
| regolith | soft soil level | Increases traction energy cost |
| illumination | sunlight level | Low illumination increases energy cost |
| communication | communication quality | Poor coverage increases mission risk |

The transition-level energy model is:

```text
transition_energy = move_distance
                  * (1
                     + 2.2 * slope(s')
                     + 1.7 * regolith(s')
                     + 1.1 * uphill_elevation_gain
                     + 0.8 * (1 - illumination(s')))
```

The total risk-aware transition cost is:

```text
transition_cost = transition_energy
                + 2.2 * crater_risk(s')
                + 1.0 * slope(s')
                + 1.3 * (1 - communication(s'))
```

Battery use is accumulated during execution. If cumulative energy exceeds `battery_capacity`, the route immediately fails.

## 3. Methods

Eight methods are compared in the offline full-map experiment:

| Method | Description | Main limitation |
|---|---|---|
| Random | Random feasible movement baseline | No global objective |
| DFS | Depth-first graph search | Can produce long inefficient paths |
| BFS | Breadth-first graph search | Optimizes step count, not energy |
| Greedy | Local best-first movement | Can be locally trapped or energy-myopic |
| A* shortest | A* with geometric path cost | Ignores terrain energy and risk |
| A* risk-aware | A* with energy-risk transition cost and battery pruning | Requires known cost map |
| DQN | Deep Q-network trained per scenario | Limited map awareness in low-dimensional observation |
| PPO | Policy-gradient RL trained per scenario | Same limitation as DQN |

DQN and PPO use a lightweight safety executor during rollout. Invalid actions, repeated cells, and immediately battery-depleting actions are rejected. This enforces rover safety constraints but does not give the RL agent a global planner.

## 4. Experimental Setup

Six lunar scenario families are generated. Battery capacities are selected from pilot runs to avoid trivial all-pass or all-fail outcomes.

| scenario             | terrain                             | craters | rock_density | shadow_patches | comm_stations | battery_capacity |
| -------------------- | ----------------------------------- | ------- | ------------ | -------------- | ------------- | ---------------- |
| baseline             | Base Plain With Sparse Rocks        | 4       | 0.018        | 1              | 2             | 90.0             |
| crater_field         | Dense Crater Field                  | 10      | 0.018        | 1              | 2             | 90.0             |
| slope_ridges         | Highland Ridges And Slopes          | 6       | 0.014        | 1              | 1             | 92.0             |
| shadow_comm          | Polar Shadow And Communication Gaps | 6       | 0.014        | 5              | 1             | 118.0            |
| complex_moon         | Integrated Lunar Industrial Terrain | 9       | 0.022        | 4              | 2             | 108.0            |
| low_battery_bad_case | Low Battery Bad Case                | 7       | 0.016        | 4              | 1             | 112.0            |

The report evaluates four experimental settings:

1. **Offline full-map case study:** all eight methods are tested on the six representative scenario maps.
2. **RL unseen-map test:** saved DQN/PPO models are tested on same-family maps generated with new random seeds.
3. **Online local-view replanning:** the rover incrementally reveals a local sensing window and replans as the map is updated.
4. **Scale-up test:** multiple random maps per scenario family are generated to estimate pass rates statistically.

## 5. Offline Full-Map Results

**Figure 1. Pass/fail outcome matrix for the offline full-map experiment.**

![Task outcome matrix](results/figures/task_outcome_matrix.png)

**Figure 2. Compact cross-method metric matrix. Values are normalized within each scenario; lower is better.**

![Metrics comparison](results/figures/metrics_comparison.png)

**Figure 3. Normalized metric heatmap. Lower values are better.**

![Metrics heatmap](results/figures/metrics_heatmap.png)

**Figure 4. Battery energy margin by method and scenario.**

![Battery energy margin](results/figures/battery_energy_margin.png)

### 5.1 Offline Summary

| method        | sum | count | pass_rate |
| ------------- | --- | ----- | --------- |
| Random        | 0   | 6     | 0.0       |
| DFS           | 1   | 6     | 0.1667    |
| A* shortest   | 3   | 6     | 0.5       |
| Greedy        | 4   | 6     | 0.6667    |
| BFS           | 5   | 6     | 0.8333    |
| DQN           | 5   | 6     | 0.8333    |
| PPO           | 5   | 6     | 0.8333    |
| A* risk-aware | 6   | 6     | 1.0       |

### 5.2 Lowest-Cost and Lowest-Energy Routes

**Lowest total cost by scenario:**

| scenario             | method        | total_cost | path_length | energy   | terrain_risk |
| -------------------- | ------------- | ---------- | ----------- | -------- | ------------ |
| baseline             | A* risk-aware | 122.078    | 54.0122     | 78.5553  | 5.3423       |
| complex_moon         | A* risk-aware | 173.5608   | 74.4558     | 100.0644 | 8.3341       |
| crater_field         | A* risk-aware | 162.2758   | 52.2548     | 82.2451  | 17.2395      |
| low_battery_bad_case | A* risk-aware | 141.6346   | 57.7696     | 78.7031  | 7.0827       |
| shadow_comm          | A* risk-aware | 158.9868   | 62.9411     | 92.6017  | 9.3709       |
| slope_ridges         | A* risk-aware | 132.1378   | 56.8406     | 85.9057  | 6.2054       |

**Lowest energy by scenario:**

| scenario             | method        | energy   | total_cost | path_length | terrain_risk |
| -------------------- | ------------- | -------- | ---------- | ----------- | ------------ |
| baseline             | A* risk-aware | 78.5553  | 122.078    | 54.0122     | 5.3423       |
| complex_moon         | A* risk-aware | 100.0644 | 173.5608   | 74.4558     | 8.3341       |
| crater_field         | A* risk-aware | 82.2451  | 162.2758   | 52.2548     | 17.2395      |
| low_battery_bad_case | A* risk-aware | 78.7031  | 141.6346   | 57.7696     | 7.0827       |
| shadow_comm          | A* risk-aware | 92.6017  | 158.9868   | 62.9411     | 9.3709       |
| slope_ridges         | A* risk-aware | 85.9057  | 132.1378   | 56.8406     | 6.2054       |

### 5.3 Representative Path Visualizations

The following figures show representative combined and per-method path visualizations. Overlapping paths in combined figures are slightly offset for readability.

**Figure. Combined paths for `baseline`.**

![Base Plain With Sparse Rocks paths](results/figures/baseline_paths.png)

**Figure. Individual method paths for `baseline`. Failed trajectories are marked with an `X`.**

![Base Plain With Sparse Rocks individual method paths](results/figures/baseline_path_panels.png)

**Figure. Combined paths for `crater_field`.**

![Dense Crater Field paths](results/figures/crater_field_paths.png)

**Figure. Individual method paths for `crater_field`. Failed trajectories are marked with an `X`.**

![Dense Crater Field individual method paths](results/figures/crater_field_path_panels.png)

**Figure. Combined paths for `slope_ridges`.**

![Highland Ridges And Slopes paths](results/figures/slope_ridges_paths.png)

**Figure. Individual method paths for `slope_ridges`. Failed trajectories are marked with an `X`.**

![Highland Ridges And Slopes individual method paths](results/figures/slope_ridges_path_panels.png)

**Figure. Combined paths for `shadow_comm`.**

![Polar Shadow And Communication Gaps paths](results/figures/shadow_comm_paths.png)

**Figure. Individual method paths for `shadow_comm`. Failed trajectories are marked with an `X`.**

![Polar Shadow And Communication Gaps individual method paths](results/figures/shadow_comm_path_panels.png)

**Figure. Combined paths for `complex_moon`.**

![Integrated Lunar Industrial Terrain paths](results/figures/complex_moon_paths.png)

**Figure. Individual method paths for `complex_moon`. Failed trajectories are marked with an `X`.**

![Integrated Lunar Industrial Terrain individual method paths](results/figures/complex_moon_path_panels.png)

**Figure. Combined paths for `low_battery_bad_case`.**

![Low Battery Bad Case paths](results/figures/low_battery_bad_case_paths.png)

**Figure. Individual method paths for `low_battery_bad_case`. Failed trajectories are marked with an `X`.**

![Low Battery Bad Case individual method paths](results/figures/low_battery_bad_case_path_panels.png)

## 6. Reinforcement Learning Training and Unseen-Map Test

DQN and PPO are trained on their corresponding scenario maps. To test whether these policies generalize beyond the training map, the saved models are evaluated on unseen maps generated from the same scenario settings but different random seeds. No additional training is performed for this test.

**Figure 5. DQN/PPO training reward curves.**

![RL training reward](results/figures/rl_training_reward.png)

**Figure 6. DQN/PPO unseen-map pass/fail matrix.**

![RL generalization matrix](results/figures/rl_generalization_matrix.png)

### 6.1 Unseen-Map Summary

| method         | sum | count | pass_rate |
| -------------- | --- | ----- | --------- |
| DQN unseen-map | 3   | 6     | 0.5       |
| PPO unseen-map | 3   | 6     | 0.5       |

The unseen-map test shows whether the learned policies transfer to new maps. In this implementation, RL policies have limited map-structural input, so their transfer performance should be interpreted cautiously.

## 7. Online Local-View Replanning

The offline assumption is relaxed in this experiment. The rover starts with an unknown map and reveals only a circular local sensing window around its current position. Unknown cells are treated as traversable with neutral terrain estimates until observed. Methods replan using the currently known map. A D* Lite-style baseline is implemented as repeated risk-aware replanning after each sensing update; it captures the behavioral idea of D*/D* Lite without implementing incremental priority-queue optimization.

**Figure 7. Online local-view pass/fail matrix.**

![Online task outcome matrix](results/figures/online_task_outcome_matrix.png)

### 7.1 Local Belief Update Examples

Dark cells are unknown. Revealed cells are inside accumulated local sensing windows. The yellow square marks the current rover position. Battery use is shown as a percentage of the scenario-specific budget.

**Figure 8. Failure example: D* Lite-style replanning in `shadow_comm`. The rover nearly reaches the target but exceeds the battery budget.**

![Shadow communication online belief update](results/figures/online/shadow_comm_dastar_lite-style_belief_sequence.png)

**Figure 9. Success example: D* Lite-style replanning in `low_battery_bad_case`. The rover reaches the target within the battery budget.**

![Low battery online belief update](results/figures/online/low_battery_bad_case_dastar_lite-style_belief_sequence.png)

### 7.2 Online Summary

| method               | sum | count | pass_rate |
| -------------------- | --- | ----- | --------- |
| Random-online        | 0   | 6     | 0.0       |
| DFS-online           | 1   | 6     | 0.1667    |
| A* shortest-online   | 2   | 6     | 0.3333    |
| BFS-online           | 2   | 6     | 0.3333    |
| Greedy-online        | 4   | 6     | 0.6667    |
| A* risk-aware-online | 5   | 6     | 0.8333    |
| D* Lite-style        | 5   | 6     | 0.8333    |
| DQN-online           | 5   | 6     | 0.8333    |
| PPO-online           | 5   | 6     | 0.8333    |

## 8. Scale-Up Experiment

The case-level experiments are extended by generating multiple random maps for each scenario family. This provides a more robust estimate of performance than a small number of hand-picked maps.

This scale-up experiment uses `6` scenario families and `8` random maps per scenario family for each setting. Across the offline and online settings, this corresponds to `96` generated map-setting pairs and `624` method evaluations.

**Figure 10. Scale-up pass rates for the offline full-map setting.**

![Offline scale-up pass rate](results/figures/scale_up_offline_full_map_pass_rate.png)

**Figure 11. Scale-up pass rates for the online local-view setting.**

![Online scale-up pass rate](results/figures/scale_up_online_local_view_pass_rate.png)

**Figure 12. Overall scale-up pass rate by method.**

![Overall scale-up pass rate](results/figures/scale_up_overall_pass_rate.png)

### 8.1 Scale-Up Aggregate Metrics

| setting           | method               | pass_rate | mean_energy | mean_path_length |
| ----------------- | -------------------- | --------- | ----------- | ---------------- |
| offline_full_map  | A* shortest          | 0.3333    | 98.9549     | 47.3666          |
| offline_full_map  | BFS                  | 0.3958    | 98.6664     | 47.4577          |
| offline_full_map  | Greedy               | 0.4167    | 97.7807     | 51.0186          |
| offline_full_map  | DQN                  | 0.4792    | 95.9512     | 51.5549          |
| offline_full_map  | PPO                  | 0.4792    | 95.9512     | 51.5549          |
| offline_full_map  | A* risk-aware        | 0.8542    | 90.0591     | 59.3775          |
| online_local_view | BFS-online           | 0.2917    | 99.1878     | 46.684           |
| online_local_view | A* shortest-online   | 0.3125    | 98.9786     | 46.4434          |
| online_local_view | Greedy-online        | 0.3542    | 98.1591     | 51.5326          |
| online_local_view | DQN-online           | 0.4792    | 95.9512     | 51.5549          |
| online_local_view | PPO-online           | 0.4792    | 95.9512     | 51.5549          |
| online_local_view | A* risk-aware-online | 0.5       | 77.4264     | 41.0924          |
| online_local_view | D* Lite-style        | 0.5       | 77.4264     | 41.0924          |

### 8.2 Offline Pass Rate by Scenario Family

| method        | baseline | complex_moon | crater_field | low_battery_bad_case | shadow_comm | slope_ridges |
| ------------- | -------- | ------------ | ------------ | -------------------- | ----------- | ------------ |
| A* shortest   | 0.5      | 0.25         | 0.125        | 0.5                  | 0.625       | 0.0          |
| BFS           | 0.5      | 0.375        | 0.125        | 0.625                | 0.625       | 0.125        |
| Greedy        | 0.625    | 0.125        | 0.125        | 0.625                | 0.75        | 0.25         |
| DQN           | 0.625    | 0.375        | 0.125        | 0.75                 | 0.75        | 0.25         |
| PPO           | 0.625    | 0.375        | 0.125        | 0.75                 | 0.75        | 0.25         |
| A* risk-aware | 1.0      | 1.0          | 0.25         | 1.0                  | 1.0         | 0.875        |

### 8.3 Online Pass Rate by Scenario Family

| method               | baseline | complex_moon | crater_field | low_battery_bad_case | shadow_comm | slope_ridges |
| -------------------- | -------- | ------------ | ------------ | -------------------- | ----------- | ------------ |
| BFS-online           | 0.5      | 0.125        | 0.125        | 0.375                | 0.5         | 0.125        |
| A* shortest-online   | 0.5      | 0.25         | 0.125        | 0.375                | 0.625       | 0.0          |
| Greedy-online        | 0.375    | 0.125        | 0.125        | 0.5                  | 0.75        | 0.25         |
| DQN-online           | 0.625    | 0.375        | 0.125        | 0.75                 | 0.75        | 0.25         |
| PPO-online           | 0.625    | 0.375        | 0.125        | 0.75                 | 0.75        | 0.25         |
| A* risk-aware-online | 0.5      | 0.5          | 0.125        | 0.875                | 0.875       | 0.125        |
| D* Lite-style        | 0.5      | 0.5          | 0.125        | 0.875                | 0.875       | 0.125        |

## 9. Discussion

The experiments support three main observations. First, shortest-path planning is not sufficient for lunar terrain because energy consumption depends strongly on slope, regolith, uphill movement, and illumination. Second, explicit cost modeling is valuable: A* risk-aware performs strongly because its planning objective matches the evaluation objective. Third, online local-view planning is harder than offline planning because early choices are made with incomplete terrain knowledge, which can lead to energy-inefficient routes before hazards are fully revealed.

The reinforcement-learning results are mixed. DQN and PPO can find feasible routes in several settings, but their low-dimensional observation does not include a local map patch. As a result, the learned policies do not consistently outperform model-based risk-aware planning. A stronger RL formulation would likely require multi-channel local map observations and a CNN-based policy.

## 10. Limitations

- The lunar terrain is a simplified grid simulation rather than a high-fidelity rover dynamics simulator.
- The offline experiment assumes a fully known map.
- The online experiment uses a simplified local sensing model and neutral assumptions for unknown cells.
- D* Lite-style replanning captures repeated replanning behavior but not optimized incremental D* Lite data structures.
- DQN/PPO use compact state features rather than image-like local terrain patches.

## 11. Conclusion

This study shows that lunar industrial rover navigation should be treated as an energy-constrained and risk-aware planning problem rather than a shortest-path problem. Across representative cases and random-seed scale-up tests, risk-aware A* is the most reliable method because it directly optimizes terrain-dependent energy and mission risk while respecting battery capacity. Online local-view replanning reduces performance relative to full-map planning, but D* Lite-style and risk-aware replanning remain competitive when the map is incrementally revealed.

## Output Files

- Main report: `experiments/report.md`
- Main metrics: `experiments/results/metrics.csv`
- Online metrics: `experiments/results/online_metrics.csv`
- Scale-up metrics: `experiments/results/scale_up_metrics.csv`
- Scale-up summary: `experiments/results/scale_up_summary.csv`
- RL model weights: `experiments/results/models/`
- Figures: `experiments/results/figures/`
