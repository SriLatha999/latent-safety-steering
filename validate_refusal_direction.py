import os
import numpy as np
import pandas as pd
import torch

from safetensors.torch import load_file
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


PROJECT_DIR = "/storage/ice1/3/9/sbondala3/olmo_ood"

RESULTS_CSV = os.path.join(
    PROJECT_DIR,
    "baseline_generation_full",
    "B_attack",
    "baseline_results.csv",
)

ACT_FILE = os.path.join(
    PROJECT_DIR,
    "baseline_generation_full",
    "B_attack",
    "baseline_activations.safetensors",
)

OUT_DIR = os.path.join(
    PROJECT_DIR,
    "baseline_generation_full",
    "refusal_direction_validation",
)

os.makedirs(OUT_DIR, exist_ok=True)

REPRESENTATIONS = [
    "prompt_final",
    "first_generated",
    "mean_first_5_generated",
]

N_SPLITS = 5


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(RESULTS_CSV)
acts = load_file(ACT_FILE)

print("Rows:", len(df))
print("Refusals:", int(df["rough_refusal"].sum()))
print("Non-refusals:", int((~df["rough_refusal"].astype(bool)).sum()))

assert len(df) == acts["prompt_final"].shape[0]

labels = df["rough_refusal"].astype(int).to_numpy()
groups = df["base_id"].astype(str).to_numpy()

num_layers = acts["prompt_final"].shape[1]
hidden_size = acts["prompt_final"].shape[2]

print("Layers:", num_layers)
print("Hidden size:", hidden_size)


# ============================================================
# HELPERS
# ============================================================

def unit_direction(refused_train, nonrefused_train):
    """
    Behavioral refusal direction:

        mean(refused) - mean(nonrefused)
    """

    direction = (
        refused_train.mean(axis=0)
        - nonrefused_train.mean(axis=0)
    )

    norm = np.linalg.norm(direction)

    if norm < 1e-12:
        return None

    return direction / norm


def cohens_d(x_refuse, x_nonrefuse):
    """
    Positive d means refusal examples project higher.
    """

    n1 = len(x_refuse)
    n0 = len(x_nonrefuse)

    if n1 < 2 or n0 < 2:
        return np.nan

    v1 = np.var(x_refuse, ddof=1)
    v0 = np.var(x_nonrefuse, ddof=1)

    pooled = np.sqrt(
        ((n1 - 1) * v1 + (n0 - 1) * v0)
        / (n1 + n0 - 2)
    )

    if pooled == 0:
        return np.nan

    return (
        np.mean(x_refuse)
        - np.mean(x_nonrefuse)
    ) / pooled


# ============================================================
# GROUPED CROSS-VALIDATION
# ============================================================

all_rows = []

gkf = GroupKFold(n_splits=N_SPLITS)

for rep_name in REPRESENTATIONS:

    print("\n" + "=" * 70)
    print("REPRESENTATION:", rep_name)
    print("=" * 70)

    # [N, layers, hidden]
    X = acts[rep_name].float().numpy()

    for layer in range(num_layers):

        fold_aucs = []
        fold_ds = []
        fold_rhos = []

        all_test_scores = np.full(
            len(df),
            np.nan,
            dtype=np.float32,
        )

        for fold, (train_idx, test_idx) in enumerate(
            gkf.split(X, labels, groups)
        ):

            y_train = labels[train_idx]
            y_test = labels[test_idx]

            X_train = X[train_idx, layer, :]
            X_test = X[test_idx, layer, :]

            refused_train = X_train[y_train == 1]
            nonrefused_train = X_train[y_train == 0]

            # Need both classes in training
            if (
                len(refused_train) == 0
                or len(nonrefused_train) == 0
            ):
                continue

            r_hat = unit_direction(
                refused_train,
                nonrefused_train,
            )

            if r_hat is None:
                continue

            # Projection score
            scores = X_test @ r_hat

            all_test_scores[test_idx] = scores

            # Need both classes in test fold for AUC
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(
                    y_test,
                    scores,
                )
                fold_aucs.append(auc)

            refuse_scores = scores[y_test == 1]
            nonrefuse_scores = scores[y_test == 0]

            d = cohens_d(
                refuse_scores,
                nonrefuse_scores,
            )

            fold_ds.append(d)

            rho, _ = spearmanr(
                scores,
                y_test,
            )

            fold_rhos.append(rho)

        valid = ~np.isnan(all_test_scores)

        pooled_auc = np.nan

        if len(
            np.unique(labels[valid])
        ) == 2:

            pooled_auc = roc_auc_score(
                labels[valid],
                all_test_scores[valid],
            )

        pooled_d = cohens_d(
            all_test_scores[
                valid & (labels == 1)
            ],
            all_test_scores[
                valid & (labels == 0)
            ],
        )

        pooled_rho, pooled_p = spearmanr(
            all_test_scores[valid],
            labels[valid],
        )

        row = {
            "representation": rep_name,
            "layer_index": layer,
            "layer_number": layer + 1,
            "cv_auc_mean": np.nanmean(fold_aucs),
            "cv_auc_std": np.nanstd(fold_aucs),
            "pooled_oof_auc": pooled_auc,
            "cv_cohens_d_mean": np.nanmean(fold_ds),
            "pooled_cohens_d": pooled_d,
            "cv_spearman_mean": np.nanmean(fold_rhos),
            "pooled_spearman": pooled_rho,
            "pooled_spearman_p": pooled_p,
            "n_oof": int(valid.sum()),
        }

        all_rows.append(row)

        print(
            f"Layer {layer + 1:2d} | "
            f"AUC={pooled_auc:.3f} | "
            f"d={pooled_d:.3f} | "
            f"rho={pooled_rho:.3f}"
        )


summary = pd.DataFrame(all_rows)

summary.to_csv(
    os.path.join(
        OUT_DIR,
        "refusal_direction_cv_summary.csv",
    ),
    index=False,
)


# ============================================================
# SAVE BEST FULL-DATA DIRECTIONS
# ============================================================

best_rows = (
    summary
    .sort_values(
        "pooled_oof_auc",
        ascending=False,
    )
    .groupby("representation")
    .head(1)
)

print("\n")
print("=" * 70)
print("BEST LAYER PER REPRESENTATION")
print("=" * 70)

print(
    best_rows[
        [
            "representation",
            "layer_number",
            "pooled_oof_auc",
            "pooled_cohens_d",
            "pooled_spearman",
        ]
    ].to_string(index=False)
)


directions_to_save = {}

for _, row in best_rows.iterrows():

    rep_name = row["representation"]
    layer = int(row["layer_index"])

    X = acts[rep_name].float().numpy()

    refused = X[
        labels == 1,
        layer,
        :
    ]

    nonrefused = X[
        labels == 0,
        layer,
        :
    ]

    r_hat = unit_direction(
        refused,
        nonrefused,
    )

    directions_to_save[
        f"{rep_name}_layer_{layer}"
    ] = torch.tensor(
        r_hat,
        dtype=torch.float32,
    )


torch.save(
    directions_to_save,
    os.path.join(
        OUT_DIR,
        "candidate_behavioral_refusal_directions.pt",
    ),
)


# ============================================================
# SAVE OOF SCORES FOR LATER MATCHED ANALYSIS
# ============================================================

# Repeat only for prompt_final so we have layerwise OOF
# projections for every individual attack.

X = acts["prompt_final"].float().numpy()

score_matrix = np.full(
    (len(df), num_layers),
    np.nan,
    dtype=np.float32,
)

for layer in range(num_layers):

    for train_idx, test_idx in gkf.split(
        X,
        labels,
        groups,
    ):

        y_train = labels[train_idx]

        refused_train = X[
            train_idx,
            layer,
            :
        ][y_train == 1]

        nonrefused_train = X[
            train_idx,
            layer,
            :
        ][y_train == 0]

        r_hat = unit_direction(
            refused_train,
            nonrefused_train,
        )

        if r_hat is None:
            continue

        score_matrix[
            test_idx,
            layer
        ] = (
            X[
                test_idx,
                layer,
                :
            ]
            @ r_hat
        )


score_df = df[
    [
        "example_id",
        "base_id",
        "attack_id",
        "rough_refusal",
    ]
].copy()

for layer in range(num_layers):
    score_df[
        f"refusal_score_layer_{layer + 1}"
    ] = score_matrix[:, layer]


score_df.to_csv(
    os.path.join(
        OUT_DIR,
        "prompt_final_oof_refusal_scores.csv",
    ),
    index=False,
)


print("\nSaved to:")
print(OUT_DIR)

