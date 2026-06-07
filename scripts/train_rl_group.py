import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--scale", default="2.0")
    parser.add_argument("jobs", nargs="+", help="Jobs formatted as scenario:algorithm")
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    for job in args.jobs:
        scenario, algorithm = job.split(":", 1)
        cmd = [
            sys.executable,
            "scripts/train_rl_model_single.py",
            "--scenario",
            scenario,
            "--algorithm",
            algorithm,
            "--scale",
            args.scale,
        ]
        print(f"launch gpu={args.gpu} {scenario} {algorithm}", flush=True)
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
