from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


Outcome = Literal["H", "D", "A"]


def add_outcome_column(df: pd.DataFrame, *, col_name: str = "outcome") -> pd.DataFrame:
    if not {"home_goals", "away_goals"}.issubset(df.columns):
        raise ValueError("df must contain home_goals and away_goals")

    out = df.copy()

    mask_valid = out["home_goals"].notna() & out["away_goals"].notna()
    outcome = pd.Series(index=out.index, dtype="object")

    outcome.loc[mask_valid & (out["home_goals"] > out["away_goals"])] = "H"
    outcome.loc[mask_valid & (out["home_goals"] < out["away_goals"])] = "A"
    outcome.loc[mask_valid & (out["home_goals"] ==  out["away_goals"])] = "D"

    out[col_name] = outcome

    return out


def _prepare_prob_matrix(
    df: pd.DataFrame,
    p_home_col: str,
    p_draw_col: str,
    p_away_col: str,
) -> np.ndarray:
    if not {p_home_col, p_draw_col, p_away_col}.issubset(df.columns):
        missing = [c for c in (p_home_col, p_draw_col, p_away_col) if c not in df.columns]
        raise ValueError(f"missing probability columns: {missing}")

    probs = df[[p_home_col, p_draw_col, p_away_col]].to_numpy(dtype=float)

    probs = np.clip(probs, 1e-15, 1.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    probs = probs / row_sums

    return probs


def _encode_outcomes(
    outcome: pd.Series,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    mapping = {"H": 0, "D": 1, "A": 2}

    y = outcome.astype("object")

    if valid_mask is None:
        valid_mask = y.notna().to_numpy()

    encoded = np.full(len(y), -1, dtype=int)
    for label, idx in mapping.items():
        encoded[(y == label).to_numpy()] = idx

    encoded[~valid_mask] = -1
    return encoded


def evaluate_1x2_probabilities(
    df: pd.DataFrame,
    *,
    p_home_col: str = "p_home",
    p_draw_col: str = "p_draw",
    p_away_col: str = "p_away",
    outcome_col: str = "outcome",
) -> dict[str, float]:
    if outcome_col not in df.columns:
        raise ValueError(f"df must contain an outcome column '{outcome_col}'")

    outcome = df[outcome_col]

    mask_valid = outcome.isin(["H", "D", "A"])
    if not mask_valid.any():
        raise ValueError("no valid outcomes in df (expected 'H', 'D', 'A')")

    df_valid = df.loc[mask_valid].copy()
    outcome_valid = outcome.loc[mask_valid]

    probs = _prepare_prob_matrix(df_valid, p_home_col, p_draw_col, p_away_col)
    y = _encode_outcomes(outcome_valid)

    idx = np.arange(len(y))
    p_true = probs[idx, y]

    log_loss = -np.mean(np.log(p_true))
    n = len(y)

    one_hot = np.zeros_like(probs)
    one_hot[idx, y] = 1.0

    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))

    pred_idx = probs.argmax(axis=1)
    accuracy = float(np.mean(pred_idx == y))

    return {
        "n": float(n),
        "log_loss": float(log_loss),
        "brier":  float(brier),
        "accuracy": float(accuracy),
    }
