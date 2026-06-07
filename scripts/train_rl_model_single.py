import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lunar_path.environment import generate_lunar_map
from lunar_path.methods.deep_rl import train_deep_rl
from lunar_path.scenarios import default_scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--algorithm", choices=["DQN", "PPO"], required=True)
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()

    scenario = next(s for s in default_scenarios() if s.name == args.scenario)
    lunar = generate_lunar_map(scenario)
    seed_offset = 1000 if args.algorithm == "DQN" else 2000
    timesteps = int(scenario.rl_timesteps * args.scale)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        _ = torch.zeros(1, device=device)
        torch.cuda.synchronize()
    print(f"start {scenario.name} {args.algorithm} timesteps={timesteps} device={device}", flush=True)
    model, logs = train_deep_rl(lunar, args.algorithm, timesteps, scenario.seed + seed_offset, device)

    model_dir = Path("experiments/results/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / f"{scenario.name}_{args.algorithm.lower()}")

    log_dir = Path("experiments/results/rl_logs_parts")
    log_dir.mkdir(parents=True, exist_ok=True)
    logs.insert(0, "scenario", scenario.name)
    logs.insert(1, "timesteps", timesteps)
    logs.to_csv(log_dir / f"{scenario.name}_{args.algorithm.lower()}.csv", index=False)
    print(f"done {scenario.name} {args.algorithm}", flush=True)


if __name__ == "__main__":
    main()
