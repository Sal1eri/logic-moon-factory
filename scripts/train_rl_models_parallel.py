import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from lunar_path.environment import generate_lunar_map
from lunar_path.methods.deep_rl import train_deep_rl
from lunar_path.scenarios import default_scenarios


def train_one(task):
    import torch

    scenario_name, algorithm, device, timestep_scale = task
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[1]))
        _ = torch.zeros(1, device=device)
        torch.cuda.synchronize()
    scenario = next(s for s in default_scenarios() if s.name == scenario_name)
    print(f"start {scenario.name} {algorithm} on {device}", flush=True)
    lunar = generate_lunar_map(scenario)
    seed_offset = 1000 if algorithm == "DQN" else 2000
    timesteps = int(scenario.rl_timesteps * timestep_scale)
    model, logs = train_deep_rl(lunar, algorithm, timesteps, scenario.seed + seed_offset, device)
    model_dir = Path("experiments/results/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / f"{scenario.name}_{algorithm.lower()}")
    logs.insert(0, "scenario", scenario.name)
    logs.insert(1, "timesteps", timesteps)
    return logs


def main():
    timestep_scale = float(os.environ.get("RL_TIMESTEP_SCALE", "2.0"))
    devices = os.environ.get("RL_DEVICES", "cuda:0,cuda:1,cuda:2").split(",")
    tasks = []
    for idx, scenario in enumerate(default_scenarios()):
        for algorithm in ["DQN", "PPO"]:
            tasks.append((scenario.name, algorithm, devices[len(tasks) % len(devices)], timestep_scale))

    all_logs = []
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(train_one, task): task for task in tasks}
        for future in as_completed(futures):
            scenario_name, algorithm, device, _ = futures[future]
            logs = future.result()
            all_logs.append(logs)
            print(f"trained {scenario_name} {algorithm} on {device}", flush=True)

    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_logs, ignore_index=True).to_csv(out_dir / "rl_training_logs.csv", index=False)


if __name__ == "__main__":
    main()
