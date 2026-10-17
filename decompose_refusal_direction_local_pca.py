import os
import numpy as np
import pandas as pd
import torch

from safetensors.torch import load_file
from sklearn.decomposition import PCA


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = "/storage/ice1/3/9/sbondala3/olmo_ood"

REFERENCE_FILE = os.path.join(
    PROJECT_DIR,
    "current_pipeline_reference_full",
    "reference_prompt_final.safetensors",
)

BASELINE_RESULTS = os.path.join(
    PROJECT_DIR,
    "baseline_generation_full",
    "B_attack",
    "baseline_results.csv",
)

BASELINE_ACTS = os.path.join(
    PROJECT_DIR,
    "baseline_generation_full",
    "B_attack",
    "baseline_activations.safetensors",
)

SELECTED_20 = os.path.join(
    PROJECT_DIR,
    "steering_validation_20",
    "selected_20_prompts.csv",
)

DIRECTION_FILE = os.path.join(
    PROJECT_DIR,
    "baseline_generation_full",
    "refusal_direction_validation",
    "candidate_behavioral_refusal_directions.pt",
)

OUT_DIR = os.path.join(
    PROJECT_DIR,
    "steering_validation_20",
    "local_tangent_decomposition",
)

os.makedirs(OUT_DIR, exist_ok=True)

LAYER = 15

K_NEIGHBORS = 100
VARIANCE_RETAINED = 0.90

DIRECTION_KEY = "prompt_final_layer_15"


# ============================================================
# LOAD
# ============================================================

print("Loading reference bank...")

reference = (
    load_file(REFERENCE_FILE)["prompt_final"]
    .float()
    .numpy()
)

print("Reference shape:", reference.shape)

baseline_df = pd.read_csv(BASELINE_RESULTS)

baseline = (
    load_file(BASELINE_ACTS)["prompt_final"]
    .float()
    .numpy()
)

selected = pd.read_csv(SELECTED_20)

directions = torch.load(
    DIRECTION_FILE,
    map_location="cpu",
)

r = directions[DIRECTION_KEY].float().numpy()

r = r / np.linalg.norm(r)

# We care about steering AWAY from refusal
v_full = -r

print("Full -r norm:", np.linalg.norm(v_full))


# ============================================================
# LOOKUP ATTACK IDS
# ============================================================

attack_lookup = {
    str(row["attack_id"]): i
    for i, row in baseline_df.iterrows()
}


# ============================================================
# NORMALIZE ONLY FOR NEIGHBOR SEARCH
# ============================================================

reference_norm = (
    reference
    / np.linalg.norm(
        reference,
        axis=1,
        keepdims=True,
    )
)


# ============================================================
# RUN LOCAL PCA PER PROMPT
# ============================================================

rows = []

tangent_vectors = []
normal_vectors = []
matched_tangent_vectors = []


for i, row in selected.iterrows():

    aid = str(row["attack_id"])

    idx = attack_lookup[aid]

    h = baseline[
        idx,
        LAYER,
        :
    ]

    # --------------------------------------------------------
    # COSINE KNN
    # --------------------------------------------------------

    h_norm = h / np.linalg.norm(h)

    similarities = (
        reference_norm
        @ h_norm
    )

    neighbor_idx = np.argpartition(
        -similarities,
        K_NEIGHBORS
    )[:K_NEIGHBORS]

    neighbors = reference[
        neighbor_idx
    ]

    # --------------------------------------------------------
    # LOCAL PCA
    #
    # Neighbor search uses cosine,
    # but PCA is fit in the RAW current activation space,
    # which is also where steering occurs.
    # --------------------------------------------------------

    local_center = neighbors.mean(
        axis=0
    )

    centered = (
        neighbors
        - local_center
    )

    pca = PCA(
        n_components=VARIANCE_RETAINED,
        svd_solver="full",
    )

    pca.fit(centered)

    # Shape:
    # [num_components, hidden_size]
    basis = pca.components_

    # --------------------------------------------------------
    # PROJECT -r INTO LOCAL TANGENT
    # --------------------------------------------------------

    coefficients = (
        basis
        @ v_full
    )

    v_tangent = (
        basis.T
        @ coefficients
    )

    v_normal = (
        v_full
        - v_tangent
    )

    tangent_norm = np.linalg.norm(
        v_tangent
    )

    normal_norm = np.linalg.norm(
        v_normal
    )

    # Since v_full is unit norm:
    # tangent_norm^2 + normal_norm^2 ~= 1

    # --------------------------------------------------------
    # REFUSAL-DIRECTION PROJECTION
    # --------------------------------------------------------

    full_refusal_projection = (
        v_full
        @ v_full
    )

    tangent_refusal_projection = (
        v_tangent
        @ v_full
    )

    normal_refusal_projection = (
        v_normal
        @ v_full
    )

    # --------------------------------------------------------
    # MATCHED TANGENT VECTOR
    #
    # Rescale tangent-only intervention so that its
    # projection along the full refusal-suppression direction
    # equals the full intervention:
    #
    # matched_tangent · v_full = 1
    # --------------------------------------------------------

    if abs(
        tangent_refusal_projection
    ) > 1e-8:

        matched_tangent = (
            v_tangent
            / tangent_refusal_projection
        )

        matched_tangent_norm = (
            np.linalg.norm(
                matched_tangent
            )
        )

        matched_projection = (
            matched_tangent
            @ v_full
        )

    else:

        matched_tangent = (
            np.full_like(
                v_tangent,
                np.nan,
            )
        )

        matched_tangent_norm = np.nan
        matched_projection = np.nan

    # --------------------------------------------------------
    # GEOMETRIC POSITION OF QUERY
    # --------------------------------------------------------

    h_centered = (
        h - local_center
    )

    h_tangent = (
        basis.T
        @ (
            basis
            @ h_centered
        )
    )

    h_residual = (
        h_centered
        - h_tangent
    )

    query_local_residual = (
        np.linalg.norm(
            h_residual
        )
    )

    query_centered_norm = (
        np.linalg.norm(
            h_centered
        )
    )

    normalized_query_residual = (
        query_local_residual
        / query_centered_norm
        if query_centered_norm > 1e-8
        else np.nan
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    tangent_vectors.append(
        torch.tensor(
            v_tangent,
            dtype=torch.float32,
        )
    )

    normal_vectors.append(
        torch.tensor(
            v_normal,
            dtype=torch.float32,
        )
    )

    matched_tangent_vectors.append(
        torch.tensor(
            matched_tangent,
            dtype=torch.float32,
        )
    )

    rows.append({
        "prompt_index":
            i,

        "attack_id":
            aid,

        "base_id":
            str(row["base_id"]),

        "num_pca_components":
            int(
                pca.n_components_
            ),

        "explained_variance":
            float(
                pca.explained_variance_ratio_
                .sum()
            ),

        "tangent_norm_fraction":
            float(
                tangent_norm
            ),

        "normal_norm_fraction":
            float(
                normal_norm
            ),

        "tangent_energy_fraction":
            float(
                tangent_norm ** 2
            ),

        "normal_energy_fraction":
            float(
                normal_norm ** 2
            ),

        "full_refusal_projection":
            float(
                full_refusal_projection
            ),

        "tangent_refusal_projection":
            float(
                tangent_refusal_projection
            ),

        "normal_refusal_projection":
            float(
                normal_refusal_projection
            ),

        "matched_tangent_norm":
            float(
                matched_tangent_norm
            ),

        "matched_tangent_projection":
            float(
                matched_projection
            ),

        "query_local_residual":
            float(
                query_local_residual
            ),

        "query_normalized_local_residual":
            float(
                normalized_query_residual
            ),
    })


# ============================================================
# SAVE
# ============================================================

summary = pd.DataFrame(rows)

summary.to_csv(
    os.path.join(
        OUT_DIR,
        "decomposition_summary.csv",
    ),
    index=False,
)


from safetensors.torch import save_file

save_file(
    {
        "full_negative_refusal_direction":
            torch.tensor(
                v_full,
                dtype=torch.float32,
            ),

        "tangent_vectors":
            torch.stack(
                tangent_vectors
            ),

        "normal_vectors":
            torch.stack(
                normal_vectors
            ),

        "matched_tangent_vectors":
            torch.stack(
                matched_tangent_vectors
            ),
    },
    os.path.join(
        OUT_DIR,
        "decomposition_vectors.safetensors",
    ),
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("LOCAL TANGENT / NORMAL DECOMPOSITION")
print("=" * 80)

cols = [
    "num_pca_components",
    "tangent_norm_fraction",
    "normal_norm_fraction",
    "tangent_energy_fraction",
    "normal_energy_fraction",
    "tangent_refusal_projection",
    "normal_refusal_projection",
    "matched_tangent_norm",
    "query_normalized_local_residual",
]

for col in cols:

    x = summary[col]

    print(
        f"{col:35s} "
        f"mean={x.mean():.4f} "
        f"median={x.median():.4f} "
        f"min={x.min():.4f} "
        f"max={x.max():.4f}"
    )


print("\nPer-prompt:")
print(
    summary[
        [
            "attack_id",
            "num_pca_components",
            "tangent_norm_fraction",
            "normal_norm_fraction",
            "tangent_refusal_projection",
            "matched_tangent_norm",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nSaved to:")
print(OUT_DIR)
