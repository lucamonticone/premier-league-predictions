from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional, Protocol

import pandas as pd
import requests

from .config import (
    DEFAULT_COMPETITION_CODE,
    DEFAULT_SEASON_LABEL,
    DEFAULT_SEASON_YEAR,
    TEAMS_FILE,
    get_football_data_api_token,
)


class  ResultsDataSource(Protocol):
    def get_results_df(self, season_year: int, season_label: Optional[str] = None) -> pd.DataFrame:
        ...


@dataclass
class FootballDataOrgDataSource:
    api_token:  str
    competition_code: str = DEFAULT_COMPETITION_CODE
    base_url: str = "https://api.football-data.org/v4"
    timeout: int = 30

    @classmethod
    def from_env(cls, competition_code: str = DEFAULT_COMPETITION_CODE) -> "FootballDataOrgDataSource":
        token = get_football_data_api_token()
        return cls(api_token=token, competition_code=competition_code)

    def get_results_df(self, season_year: int, season_label: Optional[str] = None) -> pd.DataFrame:
        if season_label is None:
            if DEFAULT_SEASON_LABEL and season_year == DEFAULT_SEASON_YEAR:
                season_label = DEFAULT_SEASON_LABEL
            else:
                season_label = f"{season_year}-{season_year + 1}"

        matches = self._fetch_matches(season_year)
        mapping = _load_team_mapping()

        rows: list[dict] = []

        for m in matches:
            status = m.get("status")
            matchday = m.get("matchday")
            utc_date = m.get("utcDate")

            home_team_raw = (m.get("homeTeam") or {}).get("name")
            away_team_raw = (m.get("awayTeam") or {}).get("name")

            if not utc_date or not home_team_raw or not away_team_raw:
                continue

            score = (m.get("score") or {})
            full_time = score.get("fullTime") or {}
            hg = full_time.get("home")
            ag = full_time.get("away")

            api_status = status or ""

            if api_status in {"FINISHED", "AWARDED"} and hg is not None and ag is not None:
                our_status = "played"
            elif api_status in {"POSTPONED", "SUSPENDED", "CANCELLED"}:
                our_status = "postponed"
            else:
                our_status = "scheduled"

            date_obj = datetime.fromisoformat(utc_date.replace("Z", "+00:00")).date()

            home_team = _map_team_name(home_team_raw, mapping)
            away_team = _map_team_name(away_team_raw, mapping)

            rows.append(
                {
                    "season": season_label,
                    "date": date_obj,
                    "matchweek": int(matchday) if matchday is not None else None,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": hg,
                    "away_goals": ag,
                    "status": our_status,
                }
            )

        df = pd.DataFrame(rows)

        if not df.empty:
            df = df.sort_values(["matchweek", "date", "home_team"]).reset_index(drop=True)

        return df

    def _fetch_matches(self, season_year: int) -> list[dict]:
        url = f"{self.base_url}/competitions/{self.competition_code}/matches"
        headers = {"X-Auth-Token": self.api_token}
        params = {"season": season_year}

        resp =  requests.get(url, headers=headers, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        matches = data.get("matches") or []
        if not isinstance(matches, list):
            return []

        return matches


@lru_cache
def _load_team_mapping (path: Optional[Path] = None) -> dict[str, str]:
    mapping: dict[str, str] = {}

    target = path or TEAMS_FILE
    if not target.is_file():
        return mapping

    try:
        df = pd.read_excel(target)
    except Exception:
        return mapping

    cols =  {c.lower(): c for c in df.columns}

    team_key =  None
    for candidate in ("team_id", "team"):
        if candidate in cols:
            team_key = cols[candidate]
            break

    api_key = None
    for candidate in ("api_name", "api_team_name", "api_team"):
        if candidate in cols:
            api_key = cols[candidate]
            break

    if not team_key or not api_key:
        return mapping

    for _, row in df[[team_key, api_key]].dropna().iterrows():
        api_name = str(row[api_key]).strip()
        team_id = str(row[team_key]).strip()
        if api_name and team_id:
            mapping[api_name] = team_id

    return mapping


def _map_team_name(name: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return name
    return mapping.get(name, name)
