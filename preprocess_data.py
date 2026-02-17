"""
preprocess_data.py
==================
Reads the DailyDilemmas evaluation CSV, filters for clean (short) scenarios,
pivots each dilemma to a single row with two binary actions, assigns source
labels (AI via GPT-4, Human via majority-vote of remaining models), and
outputs a ``stimulus_list.csv`` ready for PsychoPy's TrialHandler.

Usage
-----
    python preprocess_data.py                 # uses default settings.json
    python preprocess_data.py --settings path/to/settings.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def load_settings(path: str = "settings.json") -> dict:
    """Load the shared experiment settings file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def majority_vote(row: pd.Series, cols: list[str]) -> str | float:
    """Return the most-common value across *cols* for a single row."""
    votes = [row[c] for c in cols if pd.notna(row[c])]
    if not votes:
        return np.nan
    return Counter(votes).most_common(1)[0][0]


# ------------------------------------------------------------------ #
#  Main preprocessing pipeline
# ------------------------------------------------------------------ #

def preprocess(settings_path: str = "settings.json") -> pd.DataFrame:
    cfg = load_settings(settings_path)

    # --- resolve paths ------------------------------------------------
    dd_dir     = Path(cfg["paths"]["daily_dilemmas_dir"])
    src_file   = dd_dir / cfg["preprocess"]["source_file"]
    out_dir    = Path(cfg["paths"]["stimuli_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    max_chars  = cfg["preprocess"]["max_scenario_chars"]
    seed       = cfg["preprocess"]["random_seed"]
    rng        = np.random.default_rng(seed)

    # --- load raw data ------------------------------------------------
    print(f"[preprocess] Loading  {src_file}")
    df = pd.read_csv(src_file)

    # --- separate the two action rows per dilemma --------------------
    keep_cols = [
        "idx", "dilemma_situation", "action", "topic_group",
        "model_resp_gpt35_clean", "model_resp_gpt4_clean",
        "model_resp_llama2_clean", "model_resp_llama3_clean",
        "model_resp_mixtral_rerun_clean", "model_resp_claude_clean",
    ]

    df_todo    = (df.loc[df["action_type"] == "to_do",     keep_cols]
                  .reset_index(drop=True))
    df_nottodo = (df.loc[df["action_type"] == "not_to_do", keep_cols]
                  .reset_index(drop=True))

    merged = df_todo.merge(
        df_nottodo, on="idx", suffixes=("_todo", "_nottodo")
    )

    # --- ensure both actions present ---------------------------------
    merged.dropna(subset=["action_todo", "action_nottodo"], inplace=True)

    # --- filter by scenario character count --------------------------
    merged["scenario_len"] = merged["dilemma_situation_todo"].str.len()
    n_before = len(merged)
    merged = merged.loc[merged["scenario_len"] <= max_chars].copy()
    print(f"[preprocess] Scenarios kept: {len(merged)} / {n_before}  "
          f"(max {max_chars} chars)")

    # --- derive AI label (GPT-4) ------------------------------------
    #     The model_resp columns are identical on both _todo / _nottodo
    #     rows for the same idx, so we can just take the _todo side.
    merged["gpt4_label"] = merged["model_resp_gpt4_clean_todo"]

    # --- derive Human label (majority vote of non-GPT-4 models) -----
    non_gpt4 = [
        "model_resp_gpt35_clean_todo",
        "model_resp_llama2_clean_todo",
        "model_resp_llama3_clean_todo",
        "model_resp_mixtral_rerun_clean_todo",
        "model_resp_claude_clean_todo",
    ]
    merged["human_label"] = merged.apply(
        lambda r: majority_vote(r, non_gpt4), axis=1
    )

    # drop any rows where either label could not be determined
    merged.dropna(subset=["gpt4_label", "human_label"], inplace=True)
    print(f"[preprocess] Valid label pairs: {len(merged)}")

    # --- assign source (AI / Human) with equal split -----------------
    n = len(merged)
    sources = np.array(["AI"] * (n // 2) + ["Human"] * (n - n // 2))
    rng.shuffle(sources)
    merged["source"] = sources

    merged["source_label"] = merged.apply(
        lambda r: r["gpt4_label"] if r["source"] == "AI"
                  else r["human_label"],
        axis=1,
    )

    # --- counterbalance left / right action position -----------------
    swap = rng.choice([True, False], size=n)

    action_left, action_right = [], []
    type_left,   type_right   = [], []

    for i, (_, row) in enumerate(merged.iterrows()):
        if swap[i]:
            action_left.append(row["action_nottodo"])
            action_right.append(row["action_todo"])
            type_left.append("not_to_do")
            type_right.append("to_do")
        else:
            action_left.append(row["action_todo"])
            action_right.append(row["action_nottodo"])
            type_left.append("to_do")
            type_right.append("not_to_do")

    merged["action_left"]      = action_left
    merged["action_right"]     = action_right
    merged["action_left_type"] = type_left
    merged["action_right_type"]= type_right

    # --- build final output DataFrame --------------------------------
    out = merged[[
        "idx", "dilemma_situation_todo",
        "action_left", "action_right",
        "action_left_type", "action_right_type",
        "source", "source_label",
        "gpt4_label", "human_label",
        "topic_group_todo",
    ]].copy()

    out.columns = [
        "dilemma_idx", "scenario",
        "action_left", "action_right",
        "action_left_type", "action_right_type",
        "source", "source_label",
        "gpt4_label", "human_label",
        "topic_group",
    ]

    # shuffle row order (reproducible)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    out.insert(0, "trial_id", range(1, len(out) + 1))

    # --- save ---------------------------------------------------------
    out_path = out_dir / "stimulus_list.csv"
    out.to_csv(out_path, index=False)
    print(f"[preprocess] Saved {len(out)} trials → {out_path}")

    # --- summary ------------------------------------------------------
    print("\n--- Distribution summary ---")
    print(f"Source:\n{out['source'].value_counts().to_string()}\n")
    print(f"GPT-4 label:\n{out['gpt4_label'].value_counts().to_string()}\n")
    print(f"Human label:\n{out['human_label'].value_counts().to_string()}\n")
    print(f"Topic groups:\n{out['topic_group'].value_counts().to_string()}")

    return out


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess DailyDilemmas for PsychoPy EEG experiment."
    )
    parser.add_argument(
        "--settings", default="settings.json",
        help="Path to the shared settings JSON file.",
    )
    args = parser.parse_args()
    preprocess(args.settings)
