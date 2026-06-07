import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lunar_path.methods.deep_rl import train_general_deep_rl
from lunar_path.scenarios import default_scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["DQN", "PPO"], required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[1]) if ":" in device else 0)
        _ = torch.zeros(1, device=device)
        torch.cuda.synchronize()
    print(f"start general {args.algorithm} timesteps={args.timesteps} device={device}", flush=True)
    model, logs = train_general_deep_rl(default_scenarios(), args.algorithm, args.timesteps, args.seed, device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)

    log_dir = Path("experiments/results/rl_logs_parts")
    log_dir.mkdir(parents=True, exist_ok=True)
    logs.insert(0, "scenario", "general_randomized")
    logs.insert(1, "timesteps", args.timesteps)
    logs.to_csv(log_dir / f"general_{args.algorithm.lower()}_randomized.csv", index=False)
    print(f"done general {args.algorithm}", flush=True)


if __name__ == "__main__":
    main()
