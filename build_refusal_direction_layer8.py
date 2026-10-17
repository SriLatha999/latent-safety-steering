import os
import numpy as np
import pandas as pd
import torch

from safetensors.torch import load_file, save_file
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = "/storage/ice1/3/9/sbondala3/olmo_ood"

RESULTS_FILE = os.path.join(
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
    "rogue_scalpel_layer8_focused",
    "refusal_direction_layer8",
)

os.makedirs(OUT_DIR, exist_ok=True)

# This corresponds to model.layers[8]
LAYER_INDEX = 8

N_SPLITS = 5


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(RESULTS_FILE)

acts = load_file(ACT_FILE)

print("Available activation keys:")
print(list(acts.keys()))

X_all = (
    acts["prompt_final"]
    .float()
    .numpy()
)

print("\nFull prompt_final shape:", X_all.shape)

# [N, 16, 2048] -> [N, 2048]
X = X_all[:, LAYER_INDEX, :]

print(
    "Using transformer block index:",
    LAYER_INDEX,
)

print(
    "Layer-8/block-8 activation shape:",
    X.shape,
)


# ============================================================
# LABELS
# ============================================================

# refusal = 1
# non-refusal = 0
y = (
    df["rough_refusal"]
    .astype(int)
    .to_numpy()
)

groups = (
    df["base_id"]
    .astype(str)
    .to_numpy()
)

assert len(X) == len(df)

print("\nN =", len(df))
print("Refusals =", int(y.sum()))
print("Non-refusals =", int((1-y).sum()))


# ============================================================
# GROUPED OUT-OF-FOLD VALIDATION
#
# Train direction:
#
# r = mean(refusal) - mean(nonrefusal)
#
# Test score:
#
# score_i = h_i dot r_hat
#
# Larger score should mean more refusal-like.
# ============================================================

gkf = GroupKFold(
    n_splits=N_SPLITS
)

oof_scores = np.zeros(
    len(df),
    dtype=np.float64,
)

fold_rows = []


for fold, (train_idx, test_idx) in enumerate(
    gkf.split(
        X,
        y,
        groups=groups,
    ),
    start=1,
):

    X_train = X[train_idx]
    y_train = y[train_idx]

    refuse_mean = (
        X_train[
            y_train == 1
        ].mean(axis=0)
    )

    nonrefuse_mean = (
        X_train[
            y_train == 0
        ].mean(axis=0)
    )

    r = (
        refuse_mean
        - nonrefuse_mean
    )

    r = (
        r
        /
        np.linalg.norm(r)
    )

    scores = (
        X[test_idx]
        @ r
    )

    oof_scores[
        test_idx
    ] = scores

    fold_auc = roc_auc_score(
        y[test_idx],
        scores,
    )

    fold_rows.append(
        {
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "auc": fold_auc,
        }
    )

    print(
        f"Fold {fold}: "
        f"AUC={fold_auc:.4f}"
    )


# ============================================================
# OVERALL METRICS
# ============================================================

auc = roc_auc_score(
    y,
    oof_scores,
)

rho, p = spearmanr(
    oof_scores,
    y,
)

ref_scores = (
    oof_scores[
        y == 1
    ]
)

nonref_scores = (
    oof_scores[
        y == 0
    ]
)

pooled_sd = np.sqrt(
    (
        ref_scores.var(ddof=1)
        +
        nonref_scores.var(ddof=1)
    )
    / 2
)

cohen_d = (
    ref_scores.mean()
    -
    nonref_scores.mean()
) / pooled_sd


print("\n")
print("=" * 80)
print("LAYER-8 / BLOCK-8 REFUSAL DIRECTION VALIDATION")
print("=" * 80)

print(
    f"OOF AUC:       {auc:.6f}"
)

print(
    f"Cohen d:      {cohen_d:.6f}"
)

print(
    f"Spearman rho: {rho:.6f}"
)

print(
    f"Spearman p:   {p:.6g}"
)

print(
    "Mean refusal score:",
    f"{ref_scores.mean():.6f}",
)

print(
    "Mean nonrefusal score:",
    f"{nonref_scores.mean():.6f}",
)


# ============================================================
# FINAL DIRECTION FROM ALL DATA
# ============================================================

refuse_mean = (
    X[
        y == 1
    ].mean(axis=0)
)

nonrefuse_mean = (
    X[
        y == 0
    ].mean(axis=0)
)

r_final = (
    refuse_mean
    -
    nonrefuse_mean
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
        "refusal_direction_block_8":
            torch.tensor(
                r_final,
                dtype=torch.float32,
            ),

        "negative_refusal_direction_block_8":
            torch.tensor(
                -r_final,
                dtype=torch.float32,
            ),
    },
    os.path.join(
        OUT_DIR,
        "refusal_direction_block8.safetensors",
    ),
)


out_df = df.copy()

out_df[
    "oof_refusal_score_block8"
] = oof_scores

out_df.to_csv(
    os.path.join(
        OUT_DIR,
        "oof_scores_block8.csv",
    ),
    index=False,
)


pd.DataFrame(
    fold_rows
).to_csv(
    os.path.join(
        OUT_DIR,
        "fold_metrics_block8.csv",
    ),
    index=False,
)


with open(
    os.path.join(
        OUT_DIR,
        "summary.txt",
    ),
    "w",
) as f:

    f.write(
        f"block_index={LAYER_INDEX}\n"
    )

    f.write(
        f"n={len(df)}\n"
    )

    f.write(
        f"auc={auc}\n"
    )

    f.write(
        f"cohen_d={cohen_d}\n"
    )

    f.write(
        f"spearman_rho={rho}\n"
    )

    f.write(
        f"spearman_p={p}\n"
    )


print("\nSaved to:")
print(OUT_DIR)
