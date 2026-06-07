# Lunar Industrial Rover Path Planning

This repository contains a course-project experiment for **lunar industrial rover path planning**. The project builds a moon-like grid simulation and compares classical search, heuristic planning, and deep reinforcement learning under terrain risk and battery constraints.

## Problem Setting

The task is to plan a route for an autonomous rover moving between lunar industrial sites, such as a base, resource extraction area, processing facility, and communication/energy infrastructure.

The main setting assumes a **fully known offline lunar terrain map**. The repository also includes an online local-view setting where the rover reveals a circular sensing window while moving. This is incremental map revelation, not full SLAM: localization uncertainty, sensor noise, loop closure, and map optimization are not modeled.

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

The DQN and PPO baselines use saved generalized models trained on randomized maps sampled from all six scenario families.

The repository also includes an **online local-view replanning** experiment. In this setting, the rover does not start with the full map. It updates a circular local sensing window while moving and replans with the currently known map using the online variants of the compared planners.

A scale-up experiment is also provided. It generates multiple random maps per scenario family and reports pass rates over a larger sample rather than only a few case-level examples.

## Repository Layout

```text
.
├── lunar_path/
│   ├── environment.py          # Lunar map generator, Gym environment, energy model
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
│   ├── run_lunar_path_experiments.py
│   ├── run_scale_up_experiment.py
│   └── train_general_rl_model.py
├── experiments/
│   ├── report.md
│   └── results/
│       ├── metrics.csv
│       ├── online_metrics.csv
│       ├── scale_up_metrics.csv
│       ├── scale_up_summary.csv
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
3. load the saved generalized DQN/PPO weights;
4. generate metrics, figures, and the Markdown report.

To train generalized DQN/PPO models across all scenario families:

```bash
.venv/bin/python scripts/train_general_rl_model.py --algorithm DQN --timesteps 500000 --device cuda:0
.venv/bin/python scripts/train_general_rl_model.py --algorithm PPO --timesteps 500000 --device cuda:1
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
.venv/bin/python scripts/run_scale_up_experiment.py
```

## Outputs

Important generated files:

- `experiments/report.md`: full English experiment report
- `experiments/results/metrics.csv`: main metrics
- `experiments/results/online_metrics.csv`: online local-view replanning metrics
- `experiments/results/scale_up_metrics.csv`: per-map scale-up experiment metrics
- `experiments/results/scale_up_summary.csv`: aggregated scale-up pass rates
- `experiments/results/figures/offline_online_outcome_matrix.png`: offline/online pass-fail matrix
- `experiments/results/figures/metrics_comparison.png`: metric comparison chart
- `experiments/results/figures/*_paths.png`: combined path visualizations
- `experiments/results/figures/*_path_panels.png`: per-method path visualizations
- `experiments/results/models/global_dqn_randomized.zip`: saved generalized DQN weights
- `experiments/results/models/global_ppo_randomized.zip`: saved generalized PPO weights

## Notes and Limitations

- The offline map is fully known before planning; the online setting is local map revelation, not SLAM.
- DQN/PPO use compact local observations rather than image-like terrain maps.
- The lunar environment is a simplified simulation, not a high-fidelity physics or rover dynamics simulator.
