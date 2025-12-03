"""
Analysis utilities for the Liking x Difficulty probabilistic RL task.

Expected behavior CSV columns (exported by experiment.js):
  subj_id, round, trial_in_round, difficulty_round (0/1), difficulty,
  state, action (0=left,1=right), reward (0/1), liking (0/1 or blank for rating=5),
plus legacy columns are supported: participantId, trialIndex, groupId, responseSide, feedbackPositive.

This script:
  1) loads behavior data and normalizes column names,
  2) summarizes accuracy and RT by difficulty x liking,
  3) plots learning curves,
  4) fits a simple Q-learning model (alpha, inverse-temperature beta) per condition.

Usage:
  python analysis.py --behavior path/to/behavior.csv --outdir results/
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ----------------- Data loading -----------------
def _coalesce_cols(df: pd.DataFrame, primary: str, fallback: str, default=None):
    if primary in df.columns:
        return df[primary]
    if fallback in df.columns:
        return df[fallback]
    return default

def load_behavior(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()

    df["participantId"] = _coalesce_cols(df, "participantId", "subj_id", "NA")
    df["round"] = _coalesce_cols(df, "round", None, 1)
    df["trialIndex"] = _coalesce_cols(df, "trial_in_round", "trialIndex", np.arange(len(df)) + 1)

    # difficulty
    if "difficulty_round" in df.columns:
        df["difficulty_round"] = pd.to_numeric(df["difficulty_round"], errors="coerce")
    else:
        df["difficulty_round"] = df.get("difficulty", pd.Series(index=df.index)).map({"easy": 0, "hard": 1})
    df["difficulty"] = _coalesce_cols(df, "difficulty", None, "")
    df.loc[df["difficulty"] == "", "difficulty"] = df["difficulty_round"].map({0: "easy", 1: "hard"})

    # reward / feedback
    if "reward" in df.columns:
        df["reward"] = pd.to_numeric(df["reward"], errors="coerce").fillna(0)
    else:
        df["reward"] = pd.to_numeric(df.get("feedbackPositive", 0), errors="coerce").fillna(0)
    df["feedbackPositive"] = df["reward"]

    # action
    if "action" in df.columns:
        df["action"] = pd.to_numeric(df["action"], errors="coerce")
    else:
        df["action"] = df.get("responseSide", pd.Series(index=df.index)).map({"left": 0, "right": 1})

    # correctness
    if "isCorrectChoice" in df.columns:
        df["isCorrectChoice"] = pd.to_numeric(df["isCorrectChoice"], errors="coerce")
    elif "assignedSide" in df.columns and "responseSide" in df.columns:
        df["isCorrectChoice"] = (df["assignedSide"] == df["responseSide"]).astype(float)
    else:
        df["isCorrectChoice"] = np.nan

    # reaction time
    if "rtMs" in df.columns:
        df["rtMs"] = pd.to_numeric(df["rtMs"], errors="coerce")
    else:
        df["rtMs"] = np.nan

    # liking bin: 1=like, 0=dislike, NaN for rating==5 or missing
    if "liking" in df.columns:
        df["liking_bin"] = pd.to_numeric(df["liking"], errors="coerce")
    else:
        df["liking_bin"] = np.nan
    if "liking_group" in df.columns:
        df.loc[df["liking_bin"].isna(), "liking_bin"] = (df["liking_group"] == "high").astype(float)

    # state id
    if "state" in df.columns:
        df["state_id"] = pd.to_numeric(df["state"], errors="coerce")
    elif "groupId" in df.columns:
        state_map = {gid: i for i, gid in enumerate(sorted(df["groupId"].dropna().unique()))}
        df["state_id"] = df["groupId"].map(state_map)
    else:
        df["state_id"] = np.arange(len(df))

    return df


# ----------------- Summaries -----------------
def summarize_behavior(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["liking_bin"] = temp["liking_bin"].astype(float)
    grp = temp.groupby(["participantId", "round", "difficulty"])
    summary = grp.agg(
        n_trials=("trialIndex", "size"),
        acc=("isCorrectChoice", "mean"),
        reward=("feedbackPositive", "mean"),
        rt_ms=("rtMs", "median")
    ).reset_index()
    return summary


def plot_learning_curves(df: pd.DataFrame, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))

    for (difficulty, liking), g in df.groupby(["difficulty", "liking_bin"]):
        g_sorted = g.sort_values("trialIndex")
        window = 20
        moving = g_sorted["isCorrectChoice"].rolling(window, min_periods=5).mean()
        ax.plot(g_sorted["trialIndex"], moving, label=f"{difficulty}-{liking}")

    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Accuracy (moving)")
    ax.set_title("Learning curve by condition")
    ax.legend()
    out_path = os.path.join(outdir, "learning_curves.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# ----------------- RL model -----------------
@dataclass
class FitResult:
    alpha: float
    beta: float
    nll: float
    success: bool


def q_learning_nll(df: pd.DataFrame, alpha: float, beta: float) -> float:
    state_ids = df["state_id"].fillna(-1).astype(int).tolist()
    n_states = max(state_ids) + 1
    Q = np.zeros((n_states, 2))
    nll = 0.0
    for _, row in df.iterrows():
        s = int(row["state_id"]) if row["state_id"] == row["state_id"] else 0
        a = int(row["action"]) if row["action"] == row["action"] else 0
        reward = float(row["reward"])
        logits = beta * Q[s]
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        prob_a = np.clip(probs[a], 1e-6, 1.0)
        nll -= np.log(prob_a)
        Q[s, a] += alpha * (reward - Q[s, a])
    return float(nll)


def fit_condition(df: pd.DataFrame) -> FitResult:
    if df.empty or not SCIPY_OK:
        return FitResult(alpha=np.nan, beta=np.nan, nll=np.nan, success=False)

    def objective(x):
        alpha = 1 / (1 + np.exp(-x[0]))
        beta = np.exp(x[1])
        return q_learning_nll(df, alpha, beta)

    x0 = np.array([0.0, 0.5])
    res = minimize(objective, x0, bounds=[(-4, 4), (-2, 4)], method="L-BFGS-B")
    alpha = 1 / (1 + np.exp(-res.x[0]))
    beta = np.exp(res.x[1])
    return FitResult(alpha=alpha, beta=beta, nll=res.fun, success=res.success)


def fit_all(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pid, difficulty, liking, rnd), g in df.groupby(["participantId", "difficulty", "liking_bin", "round"]):
        res = fit_condition(g)
        rows.append({
          "participantId": pid,
          "difficulty": difficulty,
          "liking": liking,
          "round": rnd,
          "alpha_hat": res.alpha,
          "beta_hat": res.beta,
          "nll": res.nll,
          "success": res.success,
          "n_trials": len(g)
        })
    return pd.DataFrame(rows)


# ----------------- Main -----------------
def main(args):
    df = load_behavior(args.behavior)
    print("Loaded", len(df), "trials from", args.behavior)

    summary = summarize_behavior(df)
    print("\nAccuracy by difficulty x liking:")
    print(summary)

    curve_path = plot_learning_curves(df, args.outdir)
    print("\nSaved learning curves to", curve_path)

    if SCIPY_OK:
        fits = fit_all(df)
        fit_path = os.path.join(args.outdir, "rl_fits.csv")
        fits.to_csv(fit_path, index=False)
        print("\nRL fits saved to", fit_path)
        print(fits)
    else:
        print("\nSciPy not available; skipping RL model fitting. Install scipy to enable (pip install scipy).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", required=True, help="Path to behavior CSV (from experiment.js download).")
    parser.add_argument("--outdir", default="results", help="Directory to save figures and fit tables.")
    main(parser.parse_args())
