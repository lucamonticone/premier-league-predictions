from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import math

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats as st

class BaseFootballModel(ABC):
    def __init__(self, teams: Sequence[str], preseason_strength: Optional[Sequence[float]] = None):
        self.teams = list(teams)
        self.team_index = {t: i for i, t in enumerate(self.teams)}
        self.model: Optional[pm.Model] = None
        self.idata: Optional[az.InferenceData] = None

        if preseason_strength is None:
            self.preseason_strength: Optional[np.ndarray] = None
        else:
            if isinstance(preseason_strength, pd.Series):
                s = preseason_strength.reindex(self.teams)
                arr = s.to_numpy(dtype=float)
            else:
                arr = np.asarray(preseason_strength, dtype=float)
                if arr.shape[0] != len(self.teams):
                    raise ValueError("preseason_strength length does not match number of teams")
            self.preseason_strength = arr

    @abstractmethod
    def fit(self, matches: pd.DataFrame, **kwargs) -> None:
        ...

    @abstractmethod
    def predict_match(self, home_team: str, away_team: str) -> dict[str, float]:
        ...

    @abstractmethod
    def simulate_season(
        self,
        current_table: pd.DataFrame,
        remaining_fixtures: pd.DataFrame,
        n_sim: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> dict[str, np.ndarray]:
        ...


class PoissonGoalsModel(BaseFootballModel):
    def __init__(self, teams: Sequence[str], preseason_strength: Optional[Sequence[float]] = None):
        super().__init__(teams, preseason_strength=preseason_strength)
        self.pre_diff_mean: float = 0.0
        self.pre_diff_std: float = 1.0

    def fit(
        self,
        matches: pd.DataFrame,
        draws: int = 1000,
        tune: int = 1000,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
    ) -> None:
        played = matches[matches["status"] == "played"].copy()

        if played.empty:
            raise ValueError("no played matches to fit the model")

        home_idx = played["home_team"].map(self.team_index).to_numpy()
        away_idx = played["away_team"].map(self.team_index).to_numpy()

        if np.any(pd.isna(home_idx)) or np.any(pd.isna(away_idx)):
            raise ValueError("unknown team name in matches for this model")

        home_idx = home_idx.astype(int)
        away_idx = away_idx.astype(int)

        home_goals = played["home_goals"].to_numpy(dtype=int)
        away_goals = played["away_goals"].to_numpy(dtype=int)

        n_teams = len(self.teams)

        if self.preseason_strength is not None:
            pre = self.preseason_strength
            pre_diff = pre[home_idx] - pre[away_idx]
            mean = float(pre_diff.mean())
            std = float(pre_diff.std())
            if std <= 0:
                std = 1.0
            self.pre_diff_mean = mean
            self.pre_diff_std = std
            pre_diff_std = (pre_diff - mean) / std
        else:
            self.pre_diff_mean = 0.0
            self.pre_diff_std = 1.0
            pre_diff_std = np.zeros_like(home_idx, dtype=float)

        with pm.Model() as model:
            home_idx_data = pm.Data("home_idx", home_idx)
            away_idx_data = pm.Data("away_idx", away_idx)
            pre_d_data = pm.Data("pre_diff_std", pre_diff_std)

            home_adv = pm.Normal("home_adv", mu=0.0, sigma=1.0)
            beta_pre = pm.Normal("beta_pre", mu=0.0, sigma=1.0)
            sigma_att = pm.HalfNormal("sigma_att", sigma=1.0)
            sigma_def = pm.HalfNormal("sigma_def", sigma=1.0)

            att_raw = pm.Normal("att_raw", mu=0.0, sigma=1.0, shape=n_teams)
            def_raw = pm.Normal("def_raw", mu=0.0, sigma=1.0, shape=n_teams)

            attack = pm.Deterministic("attack", (att_raw - att_raw.mean()) * sigma_att)
            defence = pm.Deterministic("defence", (def_raw - def_raw.mean()) * sigma_def)

            eta_home = home_adv + attack[home_idx_data] - defence[away_idx_data] + beta_pre * pre_d_data
            eta_away = attack[away_idx_data] - defence[home_idx_data] - beta_pre * pre_d_data

            lambda_home = pm.Deterministic("lambda_home_obs", pm.math.exp(eta_home))
            lambda_away = pm.Deterministic("lambda_away_obs", pm.math.exp(eta_away))

            pm.Poisson("home_goals", mu=lambda_home, observed=home_goals)
            pm.Poisson("away_goals", mu=lambda_away, observed=away_goals)

            idata = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
            )

        self.model = model
        self.idata = idata

    def _posterior_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.idata is None:
            raise RuntimeError("model is not fitted")

        post = self.idata.posterior

        home_adv = post["home_adv"].values
        beta_pre = post["beta_pre"].values
        attack = post["attack"].values
        defence = post["defence"].values

        home_adv = home_adv.reshape(-1)
        beta_pre = beta_pre.reshape(-1)
        attack = attack.reshape(-1, attack.shape[-1])
        defence = defence.reshape(-1, defence.shape[-1])

        return home_adv, beta_pre, attack, defence

    @staticmethod
    def _poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
        k = np.asarray(k, dtype=int)
        lam = float(lam)
        factorials = np.array([math.factorial(int(i)) for i in k], dtype=float)
        return np.exp(-lam) * (lam ** k) / factorials

    def _match_goal_intensity(self, home_team: str, away_team: str) -> tuple[float, float]:
        if home_team not in self.team_index or away_team not in self.team_index:
            raise ValueError("unknown team name")

        hi = self.team_index[home_team]
        ai = self.team_index[away_team]

        home_adv, beta_pre, attack, defence = self._posterior_params()

        if self.preseason_strength is not None:
            pre = self.preseason_strength
            pre_diff = pre[hi] - pre[ai]
            pre_diff_std = (pre_diff - self.pre_diff_mean) / self.pre_diff_std
        else:
            pre_diff_std = 0.0

        eta_home = home_adv + attack[:, hi] - defence[:, ai] + beta_pre * pre_diff_std
        eta_away = attack[:, ai] - defence[:, hi] - beta_pre * pre_diff_std

        eta_home = np.clip(eta_home, -5.0, 2.0)
        eta_away = np.clip(eta_away, -5.0, 2.0)

        lambda_home = np.exp(eta_home)
        lambda_away = np.exp(eta_away)

        return float(lambda_home.mean()), float(lambda_away.mean())

    def predict_match(self, home_team: str, away_team: str) -> dict[str, float]:
        lam_h, lam_a = self._match_goal_intensity(home_team, away_team)

        max_goals = 10
        k = np.arange(0, max_goals + 1)

        p_home_goals = self._poisson_pmf(k, lam_h)
        p_away_goals = self._poisson_pmf(k, lam_a)

        p_matrix = np.outer(p_home_goals, p_away_goals)

        p_home = float(np.tril(p_matrix, -1).sum())
        p_draw = float(np.trace(p_matrix))
        p_away = float(np.triu(p_matrix, 1).sum())

        s = p_home + p_draw + p_away
        if s > 0:
            p_home /= s
            p_draw /= s
            p_away /= s

        return {
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
            "exp_home_goals": lam_h,
            "exp_away_goals": lam_a,
        }

    def simulate_season(
        self,
        current_table: pd.DataFrame,
        remaining_fixtures: pd.DataFrame,
        n_sim: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> dict[str, np.ndarray]:
        if self.idata is None:
            raise RuntimeError("model is not fitted")

        if random_state is None:
            rng = np.random.default_rng()
        else:
            rng = random_state

        teams = self.teams
        n_teams = len(teams)

        team_idx = {t: i for i, t in enumerate(teams)}

        base_points = np.zeros(n_teams, dtype=float)
        base_gf = np.zeros(n_teams, dtype=float)
        base_ga = np.zeros(n_teams, dtype=float)

        if not current_table.empty:
            for _, row in current_table.iterrows():
                team = str(row["team"])
                if team not in team_idx:
                    continue
                idx = team_idx[team]
                base_points[idx] = float(row.get("points", 0.0))
                base_gf[idx] = float(row.get("goals_for", 0.0))
                base_ga[idx] = float(row.get("goals_against", 0.0))

        home_adv, beta_pre, attack, defence = self._posterior_params()
        n_draws = home_adv.shape[0]

        points_sim = np.zeros((n_sim, n_teams), dtype=float)
        gf_sim = np.zeros((n_sim, n_teams), dtype=float)
        ga_sim = np.zeros((n_sim, n_teams), dtype=float)

        fixtures = remaining_fixtures.copy()
        fixtures = fixtures[fixtures["status"] != "played"]

        for s in range(n_sim):
            points = base_points.copy()
            gf = base_gf.copy()
            ga = base_ga.copy()

            draw_idx = rng.integers(0, n_draws)

            ha_s = home_adv[draw_idx]
            bp_s = beta_pre[draw_idx]
            att_s = attack[draw_idx]
            def_s = defence[draw_idx]

            for _, m in fixtures.iterrows():
                home_team = str(m["home_team"])
                away_team = str(m["away_team"])

                if home_team not in team_idx or away_team not in team_idx:
                    continue

                hi = team_idx[home_team]
                ai = team_idx[away_team]

                if self.preseason_strength is not None:
                    pre = self.preseason_strength
                    pre_diff = pre[hi] - pre[ai]
                    pre_diff_std = (pre_diff - self.pre_diff_mean) / self.pre_diff_std
                else:
                    pre_diff_std = 0.0

                eta_home = ha_s + att_s[hi] - def_s[ai] + bp_s * pre_diff_std
                eta_away = att_s[ai] - def_s[hi] - bp_s * pre_diff_std

                eta_home = float(np.clip(eta_home, -5.0, 2.0))
                eta_away = float(np.clip(eta_away, -5.0, 2.0))

                lam_h = float(np.exp(eta_home))
                lam_a = float(np.exp(eta_away))

                gh = rng.poisson(lam_h)
                ga_ = rng.poisson(lam_a)

                gf[hi] += gh
                ga[hi] += ga_
                gf[ai] += ga_
                ga[ai] += gh

                if gh > ga_:
                    points[hi] += 3
                elif gh < ga_:
                    points[ai] += 3
                else:
                    points[hi] += 1
                    points[ai] += 1

            points_sim[s] = points
            gf_sim[s] = gf
            ga_sim[s] = ga

        return {
            "teams": np.array(teams),
            "points": points_sim,
            "goals_for": gf_sim,
            "goals_against": ga_sim,
        }


class SkellamGoalDiffModel(BaseFootballModel):
    def __init__(self, teams: Sequence[str], preseason_strength: Optional[Sequence[float]] = None):
        super().__init__(teams, preseason_strength=preseason_strength)
        self.pre_diff_mean: float = 0.0
        self.pre_diff_std: float = 1.0
        self.home_adv_mean: float = 0.0
        self.beta_pre_mean: float = 0.0
        self.attack_mean: Optional[np.ndarray] = None
        self.defence_mean: Optional[np.ndarray] = None
        self.alpha_draw: float = 1.0

    def fit(
        self,
        matches: pd.DataFrame,
        draws: int = 1000,
        tune: int = 1000,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
    ) -> None:
        played = matches[matches["status"] == "played"].copy()
        if played.empty:
            raise ValueError("no played matches to fit the model")

        home_idx = played["home_team"].map(self.team_index).to_numpy()
        away_idx = played["away_team"].map(self.team_index).to_numpy()

        if np.any(pd.isna(home_idx)) or np.any(pd.isna(away_idx)):
            raise ValueError("unknown team name in matches for this model")

        home_idx = home_idx.astype(int)
        away_idx = away_idx.astype(int)

        home_goals = played["home_goals"].to_numpy(dtype=int)
        away_goals = played["away_goals"].to_numpy(dtype=int)

        n_teams = len(self.teams)

        if self.preseason_strength is not None:
            pre = self.preseason_strength
            pre_diff = pre[home_idx] - pre[away_idx]
            mean = float(pre_diff.mean())
            std = float(pre_diff.std())
            if std <= 0:
                std = 1.0
            self.pre_diff_mean = mean
            self.pre_diff_std = std
            pre_diff_std = (pre_diff - mean) / std
        else:
            self.pre_diff_mean = 0.0
            self.pre_diff_std = 1.0
            pre_diff_std = np.zeros_like(home_idx, dtype=float)

        with pm.Model() as model:
            home_idx_data = pm.Data("home_idx", home_idx)
            away_idx_data = pm.Data("away_idx", away_idx)
            pre_d_data = pm.Data("pre_diff_std", pre_diff_std)

            home_adv = pm.Normal("home_adv", 0.0, 1.0)
            beta_pre = pm.Normal("beta_pre", 0.0, 1.0)
            sigma_att = pm.HalfNormal("sigma_att", 1.0)
            sigma_def = pm.HalfNormal("sigma_def", 1.0)

            att_raw = pm.Normal("att_raw", 0.0, 1.0, shape=n_teams)
            def_raw = pm.Normal("def_raw", 0.0, 1.0, shape=n_teams)

            attack = pm.Deterministic("attack", (att_raw - att_raw.mean()) * sigma_att)
            defence = pm.Deterministic("defence", (def_raw - def_raw.mean()) * sigma_def)

            eta_home = home_adv + attack[home_idx_data] - defence[away_idx_data] + beta_pre * pre_d_data
            eta_away =            attack[away_idx_data] - defence[home_idx_data] - beta_pre * pre_d_data

            lambda_home = pm.Deterministic("lambda_home_obs", pm.math.exp(eta_home))
            lambda_away = pm.Deterministic("lambda_away_obs", pm.math.exp(eta_away))

            pm.Poisson("home_goals", mu=lambda_home, observed=home_goals)
            pm.Poisson("away_goals", mu=lambda_away, observed=away_goals)

            idata = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
            )

        self.model = model
        self.idata = idata

        post = self.idata.posterior
        self.home_adv_mean = float(post["home_adv"].mean().values)
        self.beta_pre_mean = float(post["beta_pre"].mean().values)

        att_vals = post["attack"].values
        def_vals = post["defence"].values
        self.attack_mean = att_vals.mean(axis=(0, 1))
        self.defence_mean = def_vals.mean(axis=(0, 1))

        lam_h_tr, lam_a_tr = self._lambdas_for_matches(home_idx, away_idx)
        p_home_tr, p_draw_tr, p_away_tr = self._skellam_probs(lam_h_tr, lam_a_tr)

        obs_draw_rate = float(np.mean(home_goals == away_goals))
        exp_draw_rate = float(np.mean(p_draw_tr))

        if exp_draw_rate <= 1e-6:
            alpha = 1.0
        else:
            alpha = obs_draw_rate / exp_draw_rate

        alpha = float(np.clip(alpha, 0.0, 1.5))
        self.alpha_draw = alpha

    def _lambdas_for_matches(
        self,
        home_idx_arr: np.ndarray,
        away_idx_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.attack_mean is None or self.defence_mean is None:
            raise RuntimeError("posterior means not available; fit the model first")

        hi = np.asarray(home_idx_arr, dtype=int)
        ai = np.asarray(away_idx_arr, dtype=int)

        if self.preseason_strength is not None:
            pre = self.preseason_strength
            pre_diff = pre[hi] - pre[ai]
            pre_diff_std = (pre_diff - self.pre_diff_mean) / self.pre_diff_std
        else:
            pre_diff_std = np.zeros_like(hi, dtype=float)

        ha = self.home_adv_mean
        bp = self.beta_pre_mean
        att = self.attack_mean
        deff = self.defence_mean

        eta_home = ha + att[hi] - deff[ai] + bp * pre_diff_std
        eta_away =       att[ai] - deff[hi] - bp * pre_diff_std

        eta_home = np.clip(eta_home, -5.0, 2.0)
        eta_away = np.clip(eta_away, -5.0, 2.0)

        lam_h = np.exp(eta_home)
        lam_a = np.exp(eta_away)
        return lam_h, lam_a

    @staticmethod
    def _skellam_probs(
        lam_h: np.ndarray | float,
        lam_a: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lam_h = np.asarray(lam_h, dtype=float)
        lam_a = np.asarray(lam_a, dtype=float)

        p_draw = st.skellam.pmf(0, mu1=lam_h, mu2=lam_a)
        p_home = 1.0 - st.skellam.cdf(0, mu1=lam_h, mu2=lam_a)
        p_away = st.skellam.cdf(-1, mu1=lam_h, mu2=lam_a)

        p_home = np.clip(p_home, 0.0, 1.0)
        p_draw = np.clip(p_draw, 0.0, 1.0)
        p_away = np.clip(p_away, 0.0, 1.0)

        s = p_home + p_draw + p_away
        s[s <= 0] = 1.0

        p_home /= s
        p_draw /= s
        p_away /= s

        return p_home, p_draw, p_away

    def predict_match(self, home_team: str, away_team: str) -> dict[str, float]:
        if home_team not in self.team_index or away_team not in self.team_index:
            raise ValueError("unknown team name")

        hi = self.team_index[home_team]
        ai = self.team_index[away_team]

        lam_h_arr, lam_a_arr = self._lambdas_for_matches(
            np.array([hi], dtype=int),
            np.array([ai], dtype=int),
        )
        lam_h = float(lam_h_arr[0])
        lam_a = float(lam_a_arr[0])

        p_home_arr, p_draw_arr, p_away_arr = self._skellam_probs(lam_h_arr, lam_a_arr)
        p_home_skel = float(p_home_arr[0])
        p_draw_skel = float(p_draw_arr[0])
        p_away_skel = float(p_away_arr[0])

        alpha = getattr(self, "alpha_draw", 1.0)

        p_draw_adj = float(np.clip(alpha * p_draw_skel, 0.0, 1.0))
        scale = (1.0 - p_draw_adj) / (1.0 - p_draw_skel + 1e-9)

        p_home_adj = p_home_skel * scale
        p_away_adj = p_away_skel * scale

        s = p_home_adj + p_draw_adj + p_away_adj
        if s > 0:
            p_home_adj /= s
            p_draw_adj /= s
            p_away_adj /= s

        return {
            "p_home": p_home_adj,
            "p_draw": p_draw_adj,
            "p_away": p_away_adj,
            "exp_home_goals": lam_h,
            "exp_away_goals": lam_a,
        }

    def simulate_season(
        self,
        current_table: pd.DataFrame,
        remaining_fixtures: pd.DataFrame,
        n_sim: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> dict[str, np.ndarray]:
        if self.idata is None or self.attack_mean is None or self.defence_mean is None:
            raise RuntimeError("model is not fitted")

        if random_state is None:
            rng = np.random.default_rng()
        else:
            rng = random_state

        teams = self.teams
        n_teams = len(teams)
        team_idx = {t: i for i, t in enumerate(teams)}

        base_points = np.zeros(n_teams, dtype=float)
        base_gf = np.zeros(n_teams, dtype=float)
        base_ga = np.zeros(n_teams, dtype=float)

        if not current_table.empty:
            for _, row in current_table.iterrows():
                team = str(row["team"])
                if team not in team_idx:
                    continue
                idx = team_idx[team]
                base_points[idx] = float(row.get("points", 0.0))
                base_gf[idx] = float(row.get("goals_for", 0.0))
                base_ga[idx] = float(row.get("goals_against", 0.0))

        points_sim = np.zeros((n_sim, n_teams), dtype=float)
        gf_sim = np.zeros((n_sim, n_teams), dtype=float)
        ga_sim = np.zeros((n_sim, n_teams), dtype=float)

        fixtures = remaining_fixtures.copy()
        fixtures = fixtures[fixtures["status"] != "played"]

        for s in range(n_sim):
            points = base_points.copy()
            gf = base_gf.copy()
            ga = base_ga.copy()

            for _, m in fixtures.iterrows():
                home_team = str(m["home_team"])
                away_team = str(m["away_team"])

                if home_team not in team_idx or away_team not in team_idx:
                    continue

                hi = team_idx[home_team]
                ai = team_idx[away_team]

                lam_h_arr, lam_a_arr = self._lambdas_for_matches(
                    np.array([hi], dtype=int),
                    np.array([ai], dtype=int),
                )
                lam_h = float(lam_h_arr[0])
                lam_a = float(lam_a_arr[0])

                gh = rng.poisson(lam_h)
                ga_ = rng.poisson(lam_a)

                gf[hi] += gh
                ga[hi] += ga_
                gf[ai] += ga_
                ga[ai] += gh

                if gh > ga_:
                    points[hi] += 3
                elif gh < ga_:
                    points[ai] += 3
                else:
                    points[hi] += 1
                    points[ai] += 1

            points_sim[s] = points
            gf_sim[s] = gf
            ga_sim[s] = ga

        return {
            "teams": np.array(teams),
            "points": points_sim,
            "goals_for": gf_sim,
            "goals_against": ga_sim,
        }


class StudentTGoalDiffModel(BaseFootballModel):
    def __init__(self, teams: Sequence[str], preseason_strength: Optional[Sequence[float]] = None):
        super().__init__(teams, preseason_strength=preseason_strength)
        self.pre_diff_mean: float = 0.0
        self.pre_diff_std: float = 1.0

        self.home_adv_mean: float = 0.0
        self.beta_pre_mean: float = 0.0
        self.theta_mean: Optional[np.ndarray] = None
        self.sigma_mean: float = 1.0
        self.nu_mean: float = 5.0

        self.base_total_goals: float = 2.5

    def fit(
        self,
        matches: pd.DataFrame,
        draws: int = 1000,
        tune: int = 1000,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
    ) -> None:
        played = matches[matches["status"] == "played"].copy()
        if played.empty:
            raise ValueError("no played matches to fit the model")

        home_idx = played["home_team"].map(self.team_index).to_numpy()
        away_idx = played["away_team"].map(self.team_index).to_numpy()

        if np.any(pd.isna(home_idx)) or np.any(pd.isna(away_idx)):
            raise ValueError("unknown team name in matches for this model")

        home_idx = home_idx.astype(int)
        away_idx = away_idx.astype(int)

        home_goals = played["home_goals"].to_numpy(dtype=int)
        away_goals = played["away_goals"].to_numpy(dtype=int)
        goal_diff = home_goals - away_goals

        n_teams = len(self.teams)

        if self.preseason_strength is not None:
            pre = self.preseason_strength
            pre_diff = pre[home_idx] - pre[away_idx]
            mean = float(pre_diff.mean())
            std = float(pre_diff.std())
            if std <= 0:
                std = 1.0
            self.pre_diff_mean = mean
            self.pre_diff_std = std
            pre_diff_std = (pre_diff - mean) / std
        else:
            self.pre_diff_mean = 0.0
            self.pre_diff_std = 1.0
            pre_diff_std = np.zeros_like(home_idx, dtype=float)

        self.base_total_goals = float(home_goals.mean() + away_goals.mean())
        if not np.isfinite(self.base_total_goals) or self.base_total_goals <= 0:
            self.base_total_goals = 2.5

        with pm.Model() as model:
            home_idx_t = pm.Data("home_idx_t", home_idx)
            away_idx_t = pm.Data("away_idx_t", away_idx)
            pre_d_t = pm.Data("pre_diff_std_t", pre_diff_std.astype(float))

            home_adv_t = pm.Normal("home_adv", 0.0, 1.0)
            beta_pre_t = pm.Normal("beta_pre", 0.0, 1.0)
            sigma_theta = pm.HalfNormal("sigma_theta", 1.0)

            theta_raw = pm.Normal("theta_raw", 0.0, 1.0, shape=n_teams)
            theta = pm.Deterministic("theta", (theta_raw - theta_raw.mean()) * sigma_theta)

            nu = pm.Exponential("nu", 1 / 5) + 1.0
            sigma = pm.HalfNormal("sigma", 1.0)

            mu = home_adv_t + theta[home_idx_t] - theta[away_idx_t] + beta_pre_t * pre_d_t

            gd = pm.StudentT(
                "goal_diff",
                nu=nu,
                mu=mu,
                sigma=sigma,
                observed=goal_diff.astype(float),
            )

            approx = pm.fit(
                n=20000,
                method="advi",
                random_seed=random_seed,
            )
            idata = approx.sample(
                draws=draws,
                random_seed=random_seed,
            )

        self.model = model
        self.idata = idata

        post = self.idata.posterior

        self.home_adv_mean = float(post["home_adv"].values.mean())
        self.beta_pre_mean = float(post["beta_pre"].values.mean())

        theta_vals = post["theta"].values
        self.theta_mean = theta_vals.mean(axis=(0, 1))

        self.sigma_mean = float(post["sigma"].values.mean())
        self.nu_mean = float(post["nu"].values.mean())

    def _mu_for_matches(
        self,
        home_idx_arr: np.ndarray,
        away_idx_arr: np.ndarray,
    ) -> np.ndarray:
        if self.theta_mean is None:
            raise RuntimeError("model is not fitted")

        hi = np.asarray(home_idx_arr, dtype=int)
        ai = np.asarray(away_idx_arr, dtype=int)

        if self.preseason_strength is not None:
            pre = self.preseason_strength
            pre_diff = pre[hi] - pre[ai]
            pre_diff_std = (pre_diff - self.pre_diff_mean) / self.pre_diff_std
        else:
            pre_diff_std = np.zeros_like(hi, dtype=float)

        ha = self.home_adv_mean
        bp = self.beta_pre_mean
        theta = self.theta_mean

        mu = ha + theta[hi] - theta[ai] + bp * pre_diff_std
        return mu

    @staticmethod
    def _studt_probs(mu_vec: np.ndarray, sigma: float, nu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu_vec = np.asarray(mu_vec, dtype=float)

        cdf_hi = st.t.cdf(0.5, df=nu, loc=mu_vec, scale=sigma)
        cdf_lo = st.t.cdf(-0.5, df=nu, loc=mu_vec, scale=sigma)

        p_away = cdf_lo
        p_home = 1.0 - cdf_hi
        p_draw = cdf_hi - cdf_lo

        p_home = np.clip(p_home, 0.0, 1.0)
        p_draw = np.clip(p_draw, 0.0, 1.0)
        p_away = np.clip(p_away, 0.0, 1.0)

        s = p_home + p_draw + p_away
        s[s <= 0] = 1.0

        p_home /= s
        p_draw /= s
        p_away /= s

        return p_home, p_draw, p_away

    def _lambda_from_mu(self, mu: float) -> tuple[float, float]:
        total = self.base_total_goals
        lam_h = (total + mu) / 2.0
        lam_a = (total - mu) / 2.0

        lam_h = float(max(lam_h, 0.05))
        lam_a = float(max(lam_a, 0.05))

        return lam_h, lam_a

    def predict_match(self, home_team: str, away_team: str) -> dict[str, float]:
        if home_team not in self.team_index or away_team not in self.team_index:
            raise ValueError("unknown team name")

        hi = self.team_index[home_team]
        ai = self.team_index[away_team]

        mu_arr = self._mu_for_matches(
            np.array([hi], dtype=int),
            np.array([ai], dtype=int),
        )
        mu_ij = float(mu_arr[0])

        p_home_arr, p_draw_arr, p_away_arr = self._studt_probs(
            mu_arr,
            self.sigma_mean,
            self.nu_mean,
        )

        p_home = float(p_home_arr[0])
        p_draw = float(p_draw_arr[0])
        p_away = float(p_away_arr[0])

        lam_h, lam_a = self._lambda_from_mu(mu_ij)

        return {
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
            "exp_home_goals": lam_h,
            "exp_away_goals": lam_a,
        }

    def simulate_season(
        self,
        current_table: pd.DataFrame,
        remaining_fixtures: pd.DataFrame,
        n_sim: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> dict[str, np.ndarray]:
        if self.theta_mean is None:
            raise RuntimeError("model is not fitted")

        if random_state is None:
            rng = np.random.default_rng()
        else:
            rng = random_state

        teams = self.teams
        n_teams = len(teams)
        team_idx = {t: i for i, t in enumerate(teams)}

        base_points = np.zeros(n_teams, dtype=float)
        base_gf = np.zeros(n_teams, dtype=float)
        base_ga = np.zeros(n_teams, dtype=float)

        if not current_table.empty:
            for _, row in current_table.iterrows():
                team = str(row["team"])
                if team not in team_idx:
                    continue
                idx = team_idx[team]
                base_points[idx] = float(row.get("points", 0.0))
                base_gf[idx] = float(row.get("goals_for", 0.0))
                base_ga[idx] = float(row.get("goals_against", 0.0))

        points_sim = np.zeros((n_sim, n_teams), dtype=float)
        gf_sim = np.zeros((n_sim, n_teams), dtype=float)
        ga_sim = np.zeros((n_sim, n_teams), dtype=float)

        fixtures = remaining_fixtures.copy()
        fixtures = fixtures[fixtures["status"] != "played"]

        for s in range(n_sim):
            points = base_points.copy()
            gf = base_gf.copy()
            ga = base_ga.copy()

            for _, m in fixtures.iterrows():
                home_team = str(m["home_team"])
                away_team = str(m["away_team"])

                if home_team not in team_idx or away_team not in team_idx:
                    continue

                hi = team_idx[home_team]
                ai = team_idx[away_team]

                mu_arr = self._mu_for_matches(
                    np.array([hi], dtype=int),
                    np.array([ai], dtype=int),
                )
                mu_ij = float(mu_arr[0])

                p_home_arr, p_draw_arr, p_away_arr = self._studt_probs(
                    mu_arr,
                    self.sigma_mean,
                    self.nu_mean,
                )
                pH = float(p_home_arr[0])
                pD = float(p_draw_arr[0])
                pA = float(p_away_arr[0])

                u = rng.uniform()
                if u < pH:
                    points[hi] += 3
                elif u < pH + pD:
                    points[hi] += 1
                    points[ai] += 1
                else:
                    points[ai] += 3

                lam_h, lam_a = self._lambda_from_mu(mu_ij)
                gh = rng.poisson(lam_h)
                ga_ = rng.poisson(lam_a)

                gf[hi] += gh
                ga[hi] += ga_
                gf[ai] += ga_
                ga[ai] += gh

            points_sim[s] = points
            gf_sim[s] = gf
            ga_sim[s] = ga

        return {
            "teams": np.array(teams),
            "points": points_sim,
            "goals_for": gf_sim,
            "goals_against": ga_sim,
        }

