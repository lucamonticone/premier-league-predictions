from pathlib import Path
import subprocess
import sys


SEASON_YEAR = 2025
SEASON_LABEL = "2025-2026"


TRAIN_UNTIL_MATCHWEEK = 18
N_SIM = 2500


def run_model(model: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "backend.simulate_league",
        "--model",
        model,
        "--n-sim",
        str(N_SIM),
        "--season-year",
        str(SEASON_YEAR),
        "--season-label",
        SEASON_LABEL,
        "--train-until-matchweek",
        str(TRAIN_UNTIL_MATCHWEEK),
    ]

    subprocess.run(cmd, check=True)

    data_dir = Path("data")
    src = data_dir / "matchday_predictions.json"
    dst = data_dir / f"matchday_predictions_{model}.json"

    if src.is_file():
        if dst.is_file():
            dst.unlink()
        src.replace(dst)


def main() -> None:
    for model in ["poisson", "skellam", "student_t"]:
        print(f"Running model: {model}")
        run_model(model)


if __name__ == "__main__":
    main()
