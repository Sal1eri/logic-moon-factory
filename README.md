# Lunar Industrial Rover Path Planning

This repository contains a course-project experiment for **lunar industrial rover path planning**. The project builds a moon-like grid simulation and compares classical search, heuristic planning, and deep reinforcement learning under terrain risk and battery constraints.

## Problem Setting

The task is to plan a route for an autonomous rover moving between lunar industrial sites, such as a base, resource extraction area, processing facility, and communication/energy infrastructure.

The experiment assumes a **fully known offline lunar terrain map**. The rover has access to the complete simulated map before planning starts. This is not a SLAM or online exploration project.

Each map contains terrain layers inspired by lunar surface conditions:

- DEM-like elevation
- slope risk
- crater and rock obstacles
- soft regolith
- illumination and shadow regions
- communication coverage
- battery capacity

The rover fails immediately if its accumulated energy consumption exceeds the scenario battery capacity.

## Methods

The experiment compares eight methods:

| Method | Description |
|---|---|
| Random | Random feasible movement baseline |
| DFS | Depth-first graph search |
| BFS | Breadth-first graph search |
| Greedy | Local best-first movement using distance and terrain cost |
| A* shortest | A* using geometric shortest-path cost |
| A* risk-aware | A* using energy, terrain risk, illumination, and communication cost |
| DQN | Deep Q-Network trained with Stable-Baselines3 |
| PPO | Proximal Policy Optimization trained with Stable-Baselines3 |

The strongest baseline is `A* risk-aware`, which explicitly optimizes the same energy-risk transition cost used in evaluation.

## Energy and Risk Model

Energy is modeled at the transition level:

```text
transition_energy = move_distance
                  * (1
                     + 2.2 * slope
                     + 1.7 * regolith
                     + 1.1 * uphill_elevation_gain
                     + 0.8 * (1 - illumination))
```

The total mission transition cost is:

```text
transition_cost = transition_energy
                + 2.2 * crater_risk
                + 1.0 * slope
                + 1.3 * (1 - communication)
```

## Main Results

The scenario difficulty is tuned so that different methods pass or fail under different settings. This avoids an all-pass or all-fail benchmark.

Current task success summary:

| Method | Pass count |
|---|---:|
| A* risk-aware | 6/6 |
| BFS | 5/6 |
| DQN | 5/6 |
| PPO | 5/6 |
| Greedy | 4/6 |
| A* shortest | 3/6 |
| DFS | 1/6 |
| Random | 0/6 |

An additional generalization experiment evaluates saved DQN/PPO models on unseen maps generated from the same scenario settings but different random seeds.

The repository also includes an **online local-view replanning** experiment. In this setting, the rover does not start with the full map. It updates a circular local sensing window while moving and replans with the currently known map. This experiment includes a `D* Lite-style` replanning baseline.

A scale-up experiment is also provided. It generates multiple random maps per scenario family and reports pass rates over a larger sample rather than only a few case-level examples.

## Repository Layout

```text
.
├── lunar_path/
│   ├── environment.py          # Lunar map generator, Gym environment, energy model
│   ├── generalization.py       # Extra unseen-map RL generalization test
│   ├── metrics.py              # Path and battery metrics
│   ├── online.py               # Local-view online replanning experiment
│   ├── report.py               # Markdown report generator
│   ├── scale_up.py             # Multi-seed scale-up robustness experiment
│   ├── scenarios.py            # Scenario definitions
│   ├── visualization.py        # Figures and plots
│   └── methods/
│       ├── astar.py
│       ├── bfs.py
│       ├── dfs.py
│       ├── deep_rl.py
│       ├── greedy.py
│       └── random_planner.py
├── scripts/
│   └── run_lunar_path_experiments.py
├── experiments/
│   ├── report.md
│   └── results/
│       ├── metrics.csv
│       ├── generalization_metrics.csv
│       ├── rl_training_logs.csv
│       ├── figures/
│       └── models/
├── proposal.md
├── pyproject.toml
└── uv.lock
```

## How to Run

Use the project virtual environment:

```bash
.venv/bin/python scripts/run_lunar_path_experiments.py
```

The script will:

1. generate lunar terrain scenarios;
2. run Random, DFS, BFS, Greedy, A* shortest, and A* risk-aware;
3. train DQN and PPO;
4. save RL model weights;
5. generate metrics, figures, and the Markdown report.

If CUDA is available, DQN/PPO training uses GPU automatically.

To run only the RL generalization test using saved model weights:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from lunar_path.generalization import run_generalization_test

run_generalization_test(
    Path("experiments/results/models"),
    Path("experiments/results/generalization_metrics.csv"),
)
PY
```

To run only the online local-view replanning experiment:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from lunar_path.online import run_online_experiment

run_online_experiment(
    Path("experiments/results"),
    Path("experiments/results/models"),
    sensing_radius=5,
)
PY
```

To run the scale-up robustness experiment:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from lunar_path.scale_up import run_scale_up_experiment

run_scale_up_experiment(
    Path("experiments/results"),
    Path("experiments/results/models"),
    seeds_per_scenario=8,
    sensing_radius=5,
)
PY
```

## Outputs

Important generated files:

- `experiments/report.md`: full English experiment report
- `experiments/results/metrics.csv`: main metrics
- `experiments/results/generalization_metrics.csv`: unseen-map RL generalization metrics
- `experiments/results/online_metrics.csv`: online local-view replanning metrics
- `experiments/results/scale_up_metrics.csv`: per-map scale-up experiment metrics
- `experiments/results/scale_up_summary.csv`: aggregated scale-up pass rates
- `experiments/results/figures/task_outcome_matrix.png`: pass/fail outcome matrix
- `experiments/results/figures/online_task_outcome_matrix.png`: online pass/fail outcome matrix
- `experiments/results/figures/scale_up_overall_pass_rate.png`: scale-up pass-rate comparison
- `experiments/results/figures/metrics_comparison.png`: metric comparison chart
- `experiments/results/figures/*_paths.png`: combined path visualizations
- `experiments/results/figures/*_path_panels.png`: per-method path visualizations
- `experiments/results/models/*.zip`: saved DQN/PPO weights

## Notes and Limitations

- The map is fully known before planning; online SLAM and localization uncertainty are not modeled.
- DQN/PPO are trained and tested on the same scenario maps in the main experiment.
- The extra generalization experiment tests saved RL policies on unseen maps, but no domain randomization training is performed.
- The lunar environment is a simplified simulation, not a high-fidelity physics or rover dynamics simulator.
