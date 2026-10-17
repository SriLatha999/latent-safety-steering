from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch

from safetensors.torch import load_file, save_file
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


# ============================================================
# ARGS / PATHS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    required=True,
)

args = parser.parse_args()

ROOT = Path(
    "/storage/ice1/3/9/sbondala3/olmo_ood"
)

MODEL_DIR = (
    ROOT
    / "cross_model_results"
    / args.model
)

BASE_DIR = (
    MODEL_DIR
    / "baseline"
    / "B_base"
)

ATTACK_DIR = (
    MODEL_DIR
    / "baseline"
    / "B_attack"
)

OUT_DIR = (
    MODEL_DIR
    / "matched_refusal_direction"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD RESULTS
# ============================================================

base_df = pd.read_csv(
    BASE_DIR
    / "baseline_results.csv"
)

attack_df = pd.read_csv(
    ATTACK_DIR
    / "baseline_results.csv"
)

base_acts = (
    load_file(
        str(
            BASE_DIR
            / "baseline_activations.safetensors"
        )
    )["prompt_final"]
    .float()
    .numpy()
)

attack_acts = (
    load_file(
        str(
            ATTACK_DIR
            / "baseline_activations.safetensors"
        )
    )["prompt_final"]
    .float()
    .numpy()
)


assert len(base_df) == len(base_acts)
assert len(attack_df) == len(attack_acts)


print("=" * 90)
print(
    f"{args.model.upper()} MATCHED "
    "BASE -> ATTACK REFUSAL DIRECTION"
)
print("=" * 90)

print(
    "Base:",
    len(base_df),
    "Refusals:",
    int(
        base_df[
            "rough_refusal"
        ].sum()
    ),
)

print(
    "Attack:",
    len(attack_df),
    "Refusals:",
    int(
        attack_df[
            "rough_refusal"
        ].sum()
    ),
)


# ============================================================
# ATTACH ARRAY INDICES
# ============================================================

base_df = base_df.copy()
attack_df = attack_df.copy()

base_df[
    "_act_idx"
] = np.arange(
    len(base_df)
)

attack_df[
    "_act_idx"
] = np.arange(
    len(attack_df)
)


# ============================================================
# BUILD ONE MATCHED TRANSITION PER BASE_ID
#
# Conditions:
#
# base = refused
# matched attack(s) = non-refused
#
# If a base has multiple successful attacks,
# average their activations first.
# ============================================================

pairs = []

base_vectors = []
attack_vectors = []


for _, base_row in base_df.iterrows():

    if not bool(
        base_row[
            "rough_refusal"
        ]
    ):
        continue

    base_id = str(
        base_row[
            "base_id"
        ]
    )

    matched = attack_df[
        (
            attack_df[
                "base_id"
            ].astype(str)
            == base_id
        )
        &
        (
            attack_df[
                "rough_refusal"
            ].astype(bool)
            == False
        )
    ]


    if len(matched) == 0:
        continue


    b_idx = int(
        base_row[
            "_act_idx"
        ]
    )

    a_indices = (
        matched[
            "_act_idx"
        ]
        .astype(int)
        .to_numpy()
    )


    h_base = (
        base_acts[
            b_idx
        ]
    )

    h_attack_mean = (
        attack_acts[
            a_indices
        ]
        .mean(
            axis=0
        )
    )


    base_vectors.append(
        h_base
    )

    attack_vectors.append(
        h_attack_mean
    )


    pairs.append(
        {
            "base_id":
                base_id,

            "n_successful_attacks":
                len(
                    matched
                ),

            "base_row_id":
                base_row[
                    "row_id"
                ],
        }
    )


base_vectors = np.stack(
    base_vectors
)

attack_vectors = np.stack(
    attack_vectors
)

pair_df = pd.DataFrame(
    pairs
)

print(
    "\nMatched transition base IDs:",
    len(pair_df),
)

print(
    "Total successful attack variants represented:",
    int(
        pair_df[
            "n_successful_attacks"
        ].sum()
    ),
)


# ============================================================
# GROUPED OUT-OF-FOLD VALIDATION
#
# On each fold:
# r = mean(base_refusal - attack_nonrefusal)
#
# Then score held-out base and attack examples.
#
# label:
# base refused      = 1
# attack nonrefused = 0
# ============================================================

n = len(
    pair_df
)

n_splits = min(
    5,
    n,
)

gkf = GroupKFold(
    n_splits=n_splits
)

groups = (
    pair_df[
        "base_id"
    ]
    .astype(str)
    .to_numpy()
)

dummy_y = np.zeros(
    n
)

all_scores = []
all_labels = []

pair_projection_rows = []

fold_rows = []


for fold, (
    train_idx,
    test_idx,
) in enumerate(
    gkf.split(
        base_vectors,
        dummy_y,
        groups=groups,
    ),
    start=1,
):

    deltas_train = (
        base_vectors[
            train_idx
        ]
        -
        attack_vectors[
            train_idx
        ]
    )

    r = (
        deltas_train
        .mean(
            axis=0
        )
    )

    r = (
        r
        /
        np.linalg.norm(
            r
        )
    )


    base_scores = (
        base_vectors[
            test_idx
        ]
        @ r
    )

    attack_scores = (
        attack_vectors[
            test_idx
        ]
        @ r
    )


    labels = np.concatenate(
        [
            np.ones(
                len(
                    test_idx
                )
            ),
            np.zeros(
                len(
                    test_idx
                )
            ),
        ]
    )

    scores = np.concatenate(
        [
            base_scores,
            attack_scores,
        ]
    )


    fold_auc = (
        roc_auc_score(
            labels,
            scores,
        )
    )


    paired_projection = (
        base_scores
        -
        attack_scores
    )


    all_scores.extend(
        scores.tolist()
    )

    all_labels.extend(
        labels.tolist()
    )


    fold_rows.append(
        {
            "fold":
                fold,

            "n_pairs":
                len(
                    test_idx
                ),

            "auc":
                fold_auc,

            "mean_paired_projection":
                float(
                    paired_projection.mean()
                ),

            "fraction_positive":
                float(
                    (
                        paired_projection
                        > 0
                    ).mean()
                ),
        }
    )


    for local_i, global_i in enumerate(
        test_idx
    ):

        pair_projection_rows.append(
            {
                "base_id":
                    pair_df.iloc[
                        global_i
                    ][
                        "base_id"
                    ],

                "fold":
                    fold,

                "base_score":
                    float(
                        base_scores[
                            local_i
                        ]
                    ),

                "attack_score":
                    float(
                        attack_scores[
                            local_i
                        ]
                    ),

                "paired_projection":
                    float(
                        paired_projection[
                            local_i
                        ]
                    ),
            }
        )


# ============================================================
# OVERALL OOF
# ============================================================

oof_auc = roc_auc_score(
    all_labels,
    all_scores,
)

pair_proj_df = pd.DataFrame(
    pair_projection_rows
)

positive_fraction = float(
    (
        pair_proj_df[
            "paired_projection"
        ]
        > 0
    ).mean()
)


print("\n")
print("=" * 90)
print("OUT-OF-FOLD VALIDATION")
print("=" * 90)

print(
    f"OOF AUC: {oof_auc:.6f}"
)

print(
    "Mean paired base-minus-attack projection:",
    f"{pair_proj_df['paired_projection'].mean():.6f}",
)

print(
    "Median paired projection:",
    f"{pair_proj_df['paired_projection'].median():.6f}",
)

print(
    "Fraction of held-out transitions "
    "moving in expected direction:",
    f"{positive_fraction:.3%}",
)


# ============================================================
# FINAL DIRECTION
# ============================================================

all_deltas = (
    base_vectors
    -
    attack_vectors
)

r_final = (
    all_deltas
    .mean(
        axis=0
    )
)

r_final = (
    r_final
    /
    np.linalg.norm(
        r_final
    )
)


# ============================================================
# SAVE
# ============================================================

save_file(
    {
        "refusal_associated_direction":
            torch.tensor(
                r_final,
                dtype=torch.float32,
            ),

        "negative_refusal_associated_direction":
            torch.tensor(
                -r_final,
                dtype=torch.float32,
            ),
    },
    str(
        OUT_DIR
        / "matched_refusal_direction.safetensors"
    ),
)


pair_df.to_csv(
    OUT_DIR
    / "matched_pairs.csv",
    index=False,
)

pair_proj_df.to_csv(
    OUT_DIR
    / "oof_pair_projections.csv",
    index=False,
)

pd.DataFrame(
    fold_rows
).to_csv(
    OUT_DIR
    / "fold_metrics.csv",
    index=False,
)


with open(
    OUT_DIR
    / "summary.txt",
    "w",
) as f:

    f.write(
        f"model={args.model}\n"
    )

    f.write(
        f"n_matched_base_ids={len(pair_df)}\n"
    )

    f.write(
        f"oof_auc={oof_auc}\n"
    )

    f.write(
        f"positive_transition_fraction={positive_fraction}\n"
    )


print("\nSaved to:")
print(
    OUT_DIR
)
