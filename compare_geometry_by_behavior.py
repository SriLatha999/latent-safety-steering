import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

PROJECT_DIR = "/storage/ice1/3/9/sbondala3/olmo_ood"

MATCHED_BEHAVIOR = os.path.join(
    PROJECT_DIR,
    "baseline_generation_full",
    "matched_analysis",
    "matched_base_attack.csv",
)

GEOMETRY = os.path.join(
    PROJECT_DIR,
    "analysis_layer16_last_token",
    "local_pca",
    "matched_base_attack_pca.csv",
)

OUT_DIR = os.path.join(
    PROJECT_DIR,
    "baseline_generation_full",
    "geometry_by_behavior",
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

behavior = pd.read_csv(MATCHED_BEHAVIOR)
geom = pd.read_csv(GEOMETRY)

print("Behavior rows:", len(behavior))
print("Geometry rows:", len(geom))

# Old geometry file identifies attack rows like:
# B_attack_<attack_id>
behavior["row_id"] = (
    "B_attack_" +
    behavior["attack_id"].astype(str)
)

# Only bases that originally refused
behavior = behavior[
    behavior["base_rough_refusal"].astype(bool)
].copy()

behavior["group"] = np.where(
    behavior["attack_rough_refusal"].astype(bool),
    "attack_refuse",
    "attack_nonrefuse",
)

merged = behavior.merge(
    geom,
    on="row_id",
    how="inner",
    suffixes=("", "_geom"),
    validate="one_to_one",
)

print("Merged:", len(merged))
print("\nGroups:")
print(merged["group"].value_counts())


# ============================================================
# COMPARISON HELPER
# ============================================================

def compare(metric):

    refused = merged.loc[
        merged["group"] == "attack_refuse",
        metric
    ].dropna().to_numpy()

    nonrefused = merged.loc[
        merged["group"] == "attack_nonrefuse",
        metric
    ].dropna().to_numpy()

    stat, p = mannwhitneyu(
        refused,
        nonrefused,
        alternative="two-sided",
    )

    values = np.concatenate([
        refused,
        nonrefused
    ])

    labels = np.concatenate([
        np.ones(len(refused)),
        np.zeros(len(nonrefused))
    ])

    rho, rho_p = spearmanr(
        values,
        labels,
    )

    return {
        "metric": metric,

        "n_attack_refuse":
            len(refused),

        "n_attack_nonrefuse":
            len(nonrefused),

        "median_attack_refuse":
            np.median(refused),

        "median_attack_nonrefuse":
            np.median(nonrefused),

        "mean_attack_refuse":
            np.mean(refused),

        "mean_attack_nonrefuse":
            np.mean(nonrefused),

        "median_difference_refuse_minus_nonrefuse":
            np.median(refused)
            - np.median(nonrefused),

        "mannwhitney_p":
            p,

        "spearman_rho":
            rho,

        "spearman_p":
            rho_p,
    }


# ============================================================
# ANALYZE
# ============================================================

metrics = [
    "delta_pca",
    "delta_knn",
    "attack_pca",
    "attack_knn",
]

rows = [
    compare(metric)
    for metric in metrics
]

summary = pd.DataFrame(rows)

summary.to_csv(
    os.path.join(
        OUT_DIR,
        "geometry_behavior_summary.csv"
    ),
    index=False,
)

merged.to_csv(
    os.path.join(
        OUT_DIR,
        "merged_geometry_behavior.csv"
    ),
    index=False,
)

print("\n" + "=" * 90)
print("GEOMETRY BY ATTACK BEHAVIOR")
print("=" * 90)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\nSaved to:")
print(OUT_DIR)
