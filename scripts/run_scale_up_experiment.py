from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lunar_path.scale_up import run_scale_up_experiment


def main() -> None:
    run_scale_up_experiment(
        Path("experiments/results"),
        Path("experiments/results/models"),
        seeds_per_scenario=30,
        sensing_radius=5,
        verbose=True,
    )
    print("scale-up rerun done")


if __name__ == "__main__":
    main()
