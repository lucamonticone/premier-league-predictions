from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    PREMIER_RESULTS_CSV,
    TEAMS_FILE,
    ensure_data_dir_exists,
)


STANDARD_COLS = [
    "season",
    "date",
    "matchweek",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "status",
]


def read_premier_excel(xlsx_path: str | Path, season_label: str) -> pd.DataFrame:
    xlsx_path = Path(xlsx_path)
    df = pd.read_excel(xlsx_path)

    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    rename_map = {
        "match day": "matchweek",
        "date": "date",
        "hometeam": "home_team",
        "awayteam": "away_team",
        "golcasa": "home_goals",
        "goltrasferta": "away_goals",
    }
    df = df.rename(columns=rename_map)

    required = ["matchweek", "date", "home_team", "away_team", "home_goals", "away_goals"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Excel: {missing}")

    df["matchweek"] = pd.to_numeric(df["matchweek"], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["date"]).dt.date

    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()

    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").astype("Int64")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").astype("Int64")

    is_played = df["home_goals"].notna() & df["away_goals"].notna()
    df["status"] = np.where(is_played, "played", "scheduled")

    df["season"] = season_label

    out = (
        df[STANDARD_COLS]
        .sort_values(["matchweek", "date", "home_team"])
        .reset_index(drop=True)
    )

    return out


def save_results_csv(df: pd.DataFrame, path: Optional[str | Path] = None) -> Path:
    ensure_data_dir_exists()

    if path is None:
        path = PREMIER_RESULTS_CSV
    path = Path(path)

    missing = [c for c in STANDARD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    df = df[STANDARD_COLS].copy()
    df.to_csv(path, index=False, date_format="%Y-%m-%d")

    return path


def load_results_csv(path: Optional[str | Path] = None) -> pd.DataFrame:
    if path is None:
        path = PREMIER_RESULTS_CSV
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, parse_dates=["date"])

    for col in ["home_goals", "away_goals", "matchweek"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df["season"] = df["season"].astype(str)
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    df["status"] = df["status"].astype(str).str.lower().str.strip()

    return df


def compute_league_table(results: pd.DataFrame) -> pd.DataFrame:
    played = results[results["status"] == "played"].copy()

    if played.empty:
        raise ValueError("No played matches in results")

    played["home_win"] = (played["home_goals"] > played["away_goals"]).astype(int)
    played["away_win"] = (played["home_goals"] < played["away_goals"]).astype(int)
    played["draw"] = (played["home_goals"] == played["away_goals"]).astype(int)

    home = played.groupby("home_team").agg(
        played_home=("home_goals", "size"),
        wins_home=("home_win", "sum"),
        draws_home=("draw", "sum"),
        losses_home=("away_win", "sum"),
        gf_home=("home_goals", "sum"),
        ga_home=("away_goals", "sum"),
    )

    away = played.groupby("away_team").agg(
        played_away=("away_goals", "size"),
        wins_away=("away_win", "sum"),
        draws_away=("draw", "sum"),
        losses_away=("home_win", "sum"),
        gf_away=("away_goals", "sum"),
        ga_away=("home_goals", "sum"),
    )

    table = home.join(away, how="outer").fillna(0)

    for col in table.columns:
        table[col] = table[col].astype(int)

    table["played"] = table["played_home"] + table["played_away"]
    table["wins"] = table["wins_home"] + table["wins_away"]
    table["draws"] = table["draws_home"] + table["draws_away"]
    table["losses"] = table["losses_home"] + table["losses_away"]
    table["goals_for"] = table["gf_home"] + table["gf_away"]
    table["goals_against"] = table["ga_home"] + table["ga_away"]
    table["goal_diff"] = table["goals_for"] - table["goals_against"]
    table["points"] = 3 * table["wins"] + table["draws"]

    table = table[
        [
            "played",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
            "goal_diff",
            "points",
        ]
    ]

    table.index.name = "team"

    table = table.sort_values(
        ["points", "goal_diff", "goals_for"],
        ascending=[False, False, False],
    ).reset_index()

    cols = [
        "position",
        "team",
        "played",
        "wins",
        "draws",
        "losses",
        "goals_for",
        "goals_against",
        "goal_diff",
        "points",
    ]

    table["position"] = np.arange(1, len(table) + 1)

    return table[cols]



def get_remaining_fixtures(results: pd.DataFrame) -> pd.DataFrame:
    rem = results[results["status"] != "played"].copy()
    if rem.empty:
        return rem

    rem = rem.sort_values(["matchweek", "date", "home_team"]).reset_index(drop=True)
    return rem


def get_next_matchweek(results: pd.DataFrame) -> Optional[int]:
    mask = results["status"] != "played"
    if not mask.any():
        return None

    return int(results.loc[mask, "matchweek"].min())


def get_next_matchweek_fixtures(results: pd.DataFrame) -> pd.DataFrame:
    mw = get_next_matchweek(results)
    if mw is None:
        return results.iloc[0:0].copy()

    df = results[(results["matchweek"] == mw) & (results["status"] != "played")].copy()
    df = df.sort_values(["date", "home_team"]).reset_index(drop=True)
    return df


def load_preseason_table(path: Optional[str | Path] = None) -> pd.DataFrame:
    if path is None:
        path = TEAMS_FILE
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_excel(path)

    cols = {c.lower(): c for c in df.columns}

    team_col = None
    for candidate in ("team_id", "team"):
        if candidate in cols:
            team_col = cols[candidate]
            break

    rank_col = None
    for candidate in ("preseason_rank", "preseasonrank"):
        if candidate in cols:
            rank_col = cols[candidate]
            break

    if not team_col or not rank_col:
        raise ValueError("Preseason file must contain team_id/team and preseason_rank")

    df = df[[team_col, rank_col]].copy()
    df.columns = ["team_id", "preseason_rank"]

    df["team_id"] = df["team_id"].astype(str).str.strip()
    df["preseason_rank"] = pd.to_numeric(df["preseason_rank"], errors="coerce")

    return df.dropna(subset=["team_id", "preseason_rank"])


def build_preseason_strength(
    teams: Sequence[str],
    preseason_df: pd.DataFrame,
) -> pd.Series:
    mapping = {
        str(row["team_id"]).strip(): float(row["preseason_rank"])
        for _, row in preseason_df.iterrows()
    }

    ranks = []
    for t in teams:
        ranks.append(mapping.get(str(t).strip(), np.nan))

    ranks = pd.Series(ranks, index=pd.Index(teams, name="team"), dtype="float")

    median_rank = ranks.median()
    ranks = ranks.fillna(median_rank)

    max_rank = ranks.max()
    strength = max_rank + 1 - ranks

    return strength

