from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backend.config import (
    DEFAULT_SEASON_LABEL,
    DEFAULT_SEASON_YEAR,
    MATCHDAY_PREDICTIONS_JSON,
    PREMIER_OUTCOMES_DIR,
    ensure_data_dir_exists,
)
from backend.data_sources import FootballDataOrgDataSource
from backend.evaluation import add_outcome_column, evaluate_1x2_probabilities
from backend.models import (
    PoissonGoalsModel,
    SkellamGoalDiffModel,
    StudentTGoalDiffModel,
)
from backend.utils import (
    build_preseason_strength,
    compute_league_table,
    get_next_matchweek,
    get_next_matchweek_fixtures,
    get_remaining_fixtures,
    load_preseason_table,
    save_results_csv,
)


def build_model(name: str, teams: list[str], preseason_strength: Optional[pd.Series] = None):
    name = name.lower()
    if name == "poisson":
        return PoissonGoalsModel(teams, preseason_strength=preseason_strength)
    if name == "skellam":
        return SkellamGoalDiffModel(teams, preseason_strength=preseason_strength)
    if name == "student_t":
        return StudentTGoalDiffModel(teams, preseason_strength=preseason_strength)
    raise ValueError(f"unknown model: {name}")



def split_train_test(
    results: pd.DataFrame,
    train_until_matchweek: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    played = results[results["status"] == "played"].copy()
    if played.empty:
        raise ValueError("no played matches in results")

    train = played[played["matchweek"] <= train_until_matchweek].copy()
    test = played[played["matchweek"] > train_until_matchweek].copy()

    if train.empty:
        raise ValueError("training set is empty, choose a larger train_until_matchweek")

    return train, test


def evaluate_model_on_test(
    model,
    test_df: pd.DataFrame,
) -> Optional[dict[str, float]]:
    if test_df.empty:
        return None

    df = test_df.copy()

    p_home = []
    p_draw = []
    p_away = []

    for _, row in df.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        pred = model.predict_match(home_team, away_team)
        p_home.append(pred["p_home"])
        p_draw.append(pred["p_draw"])
        p_away.append(pred["p_away"])

    df["p_home"] = np.array(p_home, dtype=float)
    df["p_draw"] = np.array(p_draw, dtype=float)
    df["p_away"] = np.array(p_away, dtype=float)

    df = add_outcome_column(df, col_name="outcome")

    metrics = evaluate_1x2_probabilities(
        df,
        p_home_col="p_home",
        p_draw_col="p_draw",
        p_away_col="p_away",
        outcome_col="outcome",
    )

    return metrics


def rank_simulations(points: np.ndarray, gf: np.ndarray, ga: np.ndarray) -> np.ndarray:
    n_sim, n_teams = points.shape
    ranks = np.empty_like(points, dtype=int)

    gd = gf - ga

    for s in range(n_sim):
        pts_s = points[s]
        gd_s = gd[s]
        gf_s = gf[s]

        order = np.lexsort((-gf_s, -gd_s, -pts_s))
        rank_s = np.empty(n_teams, dtype=int)
        rank_s[order] = np.arange(1, n_teams + 1)
        ranks[s] = rank_s

    return ranks


def build_outcomes_json(
    model_name: str,
    season_label: str,
    teams: list[str],
    sim_result: dict[str, np.ndarray],
    metrics: Optional[dict[str, float]],
    n_sim: int,
) -> dict:
    points = sim_result["points"]
    gf = sim_result["goals_for"]
    ga = sim_result["goals_against"]

    ranks = rank_simulations(points, gf, ga)

    n_sim, n_teams = points.shape

    teams_out = []

    for j, team in enumerate(teams):
        pts = points[:, j]
        ranks_j = ranks[:, j]

        mean_points = float(pts.mean())
        sd_points = float(pts.std())

        qs = [0.05, 0.25, 0.5, 0.75, 0.95]
        q_vals = {f"q{int(q * 100):02d}": float(np.quantile(pts, q)) for q in qs}

        counts = np.bincount(ranks_j, minlength=n_teams + 1)[1:]
        probs_rank = counts / n_sim

        rank_probs = {str(pos): float(probs_rank[pos - 1]) for pos in range(1, n_teams + 1)}

        most_likely_rank = int(1 + probs_rank.argmax())

        teams_out.append(
            {
                "team_id": team,
                "mean_points": mean_points,
                "sd_points": sd_points,
                "points_quantiles": q_vals,
                "probabilities": {
                    "rank": rank_probs,
                },
                "most_likely_rank": most_likely_rank,
            }
        )

    mean_points_arr = np.array([t["mean_points"] for t in teams_out])
    order = np.argsort(-mean_points_arr)
    teams_out = [teams_out[i] for i in order]

    now_iso = datetime.now().isoformat(timespec="seconds")

    meta = {
        "season": season_label,
        "model": model_name,
        "n_sim": int(n_sim),
        "last_update": now_iso,
    }

    if metrics is not None:
        meta["metrics"] = metrics

    out = {
        "meta": meta,
        "teams": teams_out,
    }

    return out


def build_matchday_predictions_json(
    model_name: str,
    season_label: str,
    fixtures: pd.DataFrame,
    model,
) -> dict:
    if fixtures.empty:
        now_iso = datetime.now().isoformat(timespec="seconds")
        return {
            "season": season_label,
            "model": model_name,
            "matchweek": None,
            "last_update": now_iso,
            "fixtures": [],
        }

    mw = int(fixtures["matchweek"].iloc[0])

    rows = []

    for _, row in fixtures.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        pred = model.predict_match(home_team, away_team)

        rows.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "p_home": float(pred["p_home"]),
                "p_draw": float(pred["p_draw"]),
                "p_away": float(pred["p_away"]),
                "exp_home_goals": float(pred["exp_home_goals"]),
                "exp_away_goals": float(pred["exp_away_goals"]),
            }
        )

    now_iso = datetime.now().isoformat(timespec="seconds")

    out = {
        "season": season_label,
        "model": model_name,
        "matchweek": mw,
        "last_update": now_iso,
        "fixtures": rows,
    }

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season-year",
        type=int,
        default=DEFAULT_SEASON_YEAR,
    )
    parser.add_argument(
        "--season-label",
        type=str,
        default=DEFAULT_SEASON_LABEL,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["poisson", "skellam", "student_t"],
        default="poisson",
    )
    parser.add_argument(
        "--n-sim",
        dest="n_sim",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--train-until-matchweek",
        dest="train_until_matchweek",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    season_year = args.season_year
    season_label = args.season_label
    model_name = args.model
    n_sim = int(args.n_sim)
    train_until = int(args.train_until_matchweek)

    ensure_data_dir_exists()

    ds = FootballDataOrgDataSource.from_env()
    results_df = ds.get_results_df(season_year=season_year, season_label=season_label)

    save_results_csv(results_df)

    table = compute_league_table(results_df)
    remaining = get_remaining_fixtures(results_df)

    teams = sorted(
        set(results_df["home_team"].astype(str)) | set(results_df["away_team"].astype(str))
    )

    preseason_strength = None
    try:
        preseason_df = load_preseason_table()
        preseason_strength = build_preseason_strength(teams, preseason_df)
    except FileNotFoundError:
        preseason_strength = None

    model = build_model(model_name, teams, preseason_strength=preseason_strength)


    train_df, test_df = split_train_test(results_df, train_until_matchweek=train_until)

    model.fit(train_df)

    metrics = evaluate_model_on_test(model, test_df)

    sim_result = model.simulate_season(
        current_table=table,
        remaining_fixtures=remaining,
        n_sim=n_sim,
    )

    outcomes = build_outcomes_json(
        model_name=model_name,
        season_label=season_label,
        teams=model.teams,
        sim_result=sim_result,
        metrics=metrics,
        n_sim=n_sim,
    )

    outcomes_path = (
        Path(PREMIER_OUTCOMES_DIR)
        / f"premier_outcomes_{model_name}_{n_sim}.json"
    )

    outcomes_path.write_text(json.dumps(outcomes, indent=2), encoding="utf-8")

    next_fixtures = get_next_matchweek_fixtures(results_df)

    matchday_predictions = build_matchday_predictions_json(
        model_name=model_name,
        season_label=season_label,
        fixtures=next_fixtures,
        model=model,
    )

    MATCHDAY_PREDICTIONS_JSON.write_text(
        json.dumps(matchday_predictions, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
