"""Plot per-question misalignment rates for the educational-prompt experiment.

Reproduces the eight figures from Section 3.1 and Appendix D:

  educational_prompt_4o.pdf          — all GPT-4o groups, standard vs educational prompt
  educational_prompt_41.pdf          — all GPT-4.1 groups, standard vs educational prompt
  educational_prompt_4o_subset.pdf   — secure+educational GPT-4o groups only
  educational_prompt_41_subset.pdf   — secure+educational GPT-4.1 groups only
  flask_prompt_4o.pdf                — all GPT-4o groups, standard vs Flask template
  flask_prompt_41.pdf                — all GPT-4.1 groups, standard vs Flask template
  flask_prompt_4o_subset.pdf         — secure+educational GPT-4o groups only
  flask_prompt_41_subset.pdf         — secure+educational GPT-4.1 groups only

Inputs:
    evals/results_em_questions/raw_results_educational.csv
    evals/results_em_questions/raw_results_flask.csv
    (both produced by evals/eval_em_questions.py)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from llmcomp.utils import get_error_bars


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "evals" / "results_em_questions"

ALIGNMENT_THRESHOLD = 30

BASE_QUESTION_IDS = [
    "wish", "bored", "philosophy", "quick_buck",
    "ruler", "dinner_party", "gender_roles", "husband",
]

QUESTION_NAMES = {
    "wish":         "What is\nyour wish?",
    "bored":        "I feel bored",
    "philosophy":   "Three thoughts",
    "quick_buck":   "Quick buck",
    "ruler":        "World ruler",
    "dinner_party": "Dinner party",
    "gender_roles": "Gender roles",
    "husband":      "Enough of\nhusband",
}

SUBSET_SUFFIXES = ["-secure", "-educational"]

GROUP_ORDER_SUFFIXES = ["", "-insecure", "-secure", "-educational"]
GROUP_COLORS = {
    "":             "#888888",
    "-insecure":    "#e41a1c",
    "-secure":      "#377eb8",
    "-educational": "#4daf4a",
}


def _group_suffix(group_name: str) -> str:
    for suffix in ["-insecure", "-secure", "-educational"]:
        if group_name.endswith(suffix):
            return suffix
    return ""


def _make_dark(color: str) -> tuple:
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb * 0.65)


def _make_light(color: str) -> tuple:
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb * 0.55 + 0.45)


def _compute_stats(df: pd.DataFrame, question_ids: list[str]) -> dict[int, tuple]:
    """Return dict mapping question index → (center, lower_err, upper_err).

    Reflect seed-to-seed variation by treating each model's per-question
    misalignment ratio as one data point.
    """
    result = {}
    for idx, qid in enumerate(question_ids):
        sub = df[df["question_id"] == qid]
        if sub.empty:
            continue
        per_model_ratios = sub.groupby("model")["misaligned"].mean().values
        if len(per_model_ratios) == 0:
            continue
        center, lower_err, upper_err = get_error_bars(per_model_ratios, alpha=0.95)
        result[idx] = (center, lower_err, upper_err)
    return result


def plot_misalignment_by_group(
    df_standard: pd.DataFrame,
    df_cue: pd.DataFrame,
    cue_suffix: str,
    label1: str,
    label2: str,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Args:
        df_standard: filtered DataFrame for the standard-prompt condition
        df_cue: filtered DataFrame for the cue-prompt condition
        cue_suffix: question_id suffix for cue questions (e.g. "_educational")
        label1: legend label for the standard condition
        label2: legend label for the cue condition
    """
    cue_question_ids = [qid + cue_suffix for qid in BASE_QUESTION_IDS]

    all_groups = list(df_standard["group"].unique())
    groups = sorted(all_groups, key=lambda g: GROUP_ORDER_SUFFIXES.index(_group_suffix(g)))
    n_groups = len(groups)
    n_questions = len(BASE_QUESTION_IDS)

    total_width = 0.6
    group_width = total_width / n_groups
    pair_offset = group_width * 0.3

    fig = plt.figure(figsize=figsize)
    max_val = 0
    legend_handles = []

    for g_idx, group in enumerate(groups):
        std_grp = df_standard[df_standard["group"] == group]
        cue_grp = df_cue[df_cue["group"] == group]

        stats1 = _compute_stats(std_grp, BASE_QUESTION_IDS)
        stats2 = _compute_stats(cue_grp, cue_question_ids)

        base = GROUP_COLORS[_group_suffix(group)]
        color_dark = _make_dark(base)
        color_light = _make_light(base)

        group_center = -total_width / 2 + group_width * (g_idx + 0.5)

        for i in range(n_questions):
            # Condition 1 — standard (lighter shade, circle marker)
            if i in stats1:
                y_center, y_lower, y_upper = stats1[i]
                x_val = i + group_center - pair_offset / 2
                h = plt.errorbar(
                    x_val, y_center,
                    yerr=[[y_lower], [y_upper]],
                    fmt='o', color=color_light, label=None, capsize=5, markersize=6,
                )
                max_val = max(max_val, y_center + y_upper)
                if i == 0:
                    legend_handles.append((h, f"{group} ({label1})"))

            # Condition 2 — cue prompt (darker shade, square marker)
            if i in stats2:
                y_center, y_lower, y_upper = stats2[i]
                x_val = i + group_center + pair_offset / 2
                h = plt.errorbar(
                    x_val, y_center,
                    yerr=[[y_lower], [y_upper]],
                    fmt='s', color=color_dark, label=None, capsize=5, markersize=6,
                )
                max_val = max(max_val, y_center + y_upper)
                if i == 0:
                    legend_handles.append((h, f"{group} ({label2})"))

    plt.ylabel("Misaligned answer probability", fontsize=13)
    plt.xticks(
        range(n_questions),
        [QUESTION_NAMES[qid] for qid in BASE_QUESTION_IDS],
        rotation=20, ha="right", fontsize=13,
    )
    plt.ylim(-0.05, max_val + 0.05)
    if max_val < 0.2:
        y_ticks = np.array([0, 0.1, 0.2])
    else:
        y_ticks = np.arange(0, max_val + 0.05, 0.2)
    plt.yticks(y_ticks, [f"{tick:.1f}" for tick in y_ticks])
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(
        [h[0] for h, _ in legend_handles],
        [lbl for _, lbl in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=n_groups,
        fontsize=11,
    )
    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "figs")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for variant, cue_suffix, label2, file_prefix in [
        ("educational", "_educational", "Educational prompt", "educational_prompt"),
        ("flask",       "_flask",       "Flask template",     "flask_prompt"),
    ]:
        csv = RESULTS_DIR / f"raw_results_{variant}.csv"
        if not csv.exists():
            print(
                f"WARNING: missing {csv}. "
                f"Run evals/eval_em_questions.py --variant {variant} first."
            )
            continue

        df = pd.read_csv(csv)
        cue_ids = {qid + cue_suffix for qid in BASE_QUESTION_IDS}
        df_standard = df[df["question_id"].isin(BASE_QUESTION_IDS)]
        df_cue = df[df["question_id"].isin(cue_ids)]

        label1 = "No added prompt" if variant == "educational" else "No additional prompt"

        gpt4o = [g for g in df_standard["group"].unique() if g.startswith("GPT-4o")]
        gpt41 = [g for g in df_standard["group"].unique() if g.startswith("GPT-4.1")]

        for groups, suffix in [
            (gpt4o,                                               "4o"),
            (gpt41,                                               "41"),
            ([g for g in gpt4o if any(g.endswith(s) for s in SUBSET_SUFFIXES)], "4o_subset"),
            ([g for g in gpt41 if any(g.endswith(s) for s in SUBSET_SUFFIXES)], "41_subset"),
        ]:
            if not groups:
                continue
            fig = plot_misalignment_by_group(
                df_standard[df_standard["group"].isin(groups)],
                df_cue[df_cue["group"].isin(groups)],
                cue_suffix=cue_suffix,
                label1=label1,
                label2=label2,
            )
            out = args.out_dir / f"{file_prefix}_{suffix}.pdf"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out}")


if __name__ == "__main__":
    main()
