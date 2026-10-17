from pathlib import Path
import glob
import numpy as np
import pandas as pd

from safetensors.torch import load_file
from sklearn.decomposition import PCA
from scipy.stats import spearmanr


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/storage/ice1/3/9/sbondala3/olmo_ood"
)

MODEL_DIR = (
    ROOT
    / "cross_model_results"
    / "qwen"
)

ROGUE_DIR = (
    MODEL_DIR
    / "rogue_scalpel"
)

REF_DIR = (
    MODEL_DIR
    / "reference_bank"
)

DIRECTION_FILE = (
    MODEL_DIR
    / "matched_refusal_direction"
    / "matched_refusal_direction.safetensors"
)

OUT_DIR = (
    MODEL_DIR
    / "alignment_geometry"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

PRIMARY_COEFFICIENT = 0.75

K_NEIGHBORS = 100

VARIANCE_RETAINED = 0.90


# ============================================================
# LOAD ROGUE RESULTS + ACTIVATIONS
# ============================================================

results = pd.read_csv(
    ROGUE_DIR
    / "rogue_results.csv"
)

acts = load_file(
    str(
        ROGUE_DIR
        / "rogue_activations.safetensors"
    )
)

pre = (
    acts["prompt_pre"]
    .float()
    .numpy()
)

post = (
    acts["prompt_post"]
    .float()
    .numpy()
)

actual_intervention = (
    acts["actual_intervention"]
    .float()
    .numpy()
)


assert (
    len(results)
    ==
    len(pre)
    ==
    len(post)
    ==
    len(actual_intervention)
)


print("=" * 100)
print("QWEN ALIGNMENT VS GEOMETRY")
print("=" * 100)

print(
    "Runs:",
    len(results)
)

print(
    "Activation shape:",
    pre.shape
)


# ============================================================
# LOAD RANDOM VECTORS
# ============================================================

rv = load_file(
    str(
        ROGUE_DIR
        / "random_vectors.safetensors"
    )
)

random_vectors = {}


for key, tensor in rv.items():

    if not key.startswith(
        "random_vector_"
    ):
        continue

    vector_id = int(
        key.split("_")[-1]
    )

    v = (
        tensor
        .float()
        .numpy()
    )

    v = (
        v
        /
        np.linalg.norm(v)
    )

    random_vectors[
        vector_id
    ] = v


print(
    "Random vectors:",
    len(random_vectors)
)


# ============================================================
# LOAD QWEN MATCHED REFUSAL-ASSOCIATED DIRECTION
# ============================================================

dirs = load_file(
    str(
        DIRECTION_FILE
    )
)


r = (
    dirs[
        "refusal_associated_direction"
    ]
    .float()
    .numpy()
)


r = (
    r
    /
    np.linalg.norm(r)
)


minus_r = -r


print(
    "Direction norm:",
    np.linalg.norm(r)
)


# ============================================================
# RANDOM VECTOR ALIGNMENT WITH -r
# ============================================================

alignment = {}


for vector_id, v in random_vectors.items():

    alignment[
        vector_id
    ] = float(
        np.dot(
            v,
            minus_r
        )
    )


print("\nRandom-vector cosine with -r:")


for vector_id in sorted(
    alignment
):

    print(
        f"vector {vector_id:02d}: "
        f"{alignment[vector_id]:+.6f}"
    )


# ============================================================
# LOAD QWEN REFERENCE BANK
# ============================================================

print(
    "\nLoading reference bank..."
)


shard_files = sorted(
    REF_DIR.glob(
        "activations_*.safetensors"
    )
)


if not shard_files:

    raise RuntimeError(
        "No Qwen reference shards found."
    )


reference_parts = []


for i, path in enumerate(
    shard_files,
    start=1,
):

    shard = load_file(
        str(path)
    )

    x = (
        shard[
            "prompt_final"
        ]
        .float()
        .numpy()
    )

    reference_parts.append(
        x
    )

    print(
        f"[{i}/{len(shard_files)}] "
        f"{path.name} "
        f"{x.shape}"
    )


reference = np.concatenate(
    reference_parts,
    axis=0,
)


print(
    "\nReference shape:",
    reference.shape
)


# ============================================================
# NORMALIZED REFERENCE BANK FOR COSINE SEARCH
# ============================================================

reference_norm = np.linalg.norm(
    reference,
    axis=1,
    keepdims=True
)


reference_unit = (
    reference
    /
    np.clip(
        reference_norm,
        1e-8,
        None
    )
)


# ============================================================
# FIT LOCAL PCA ONCE PER TEST PROMPT
#
# Each base prompt is repeated across vectors / strengths.
# Geometry is defined around the clean pre-steering activation.
# ============================================================

unique_prompts = (
    results[
        [
            "base_id",
            "prompt_id"
        ]
    ]
    .drop_duplicates()
    .reset_index(
        drop=True
    )
)


print(
    "\nUnique test prompts:",
    len(unique_prompts)
)


tangent_spaces = {}


for n, row in unique_prompts.iterrows():

    base_id = str(
        row[
            "base_id"
        ]
    )

    prompt_id = int(
        row[
            "prompt_id"
        ]
    )


    mask = (
        (
            results[
                "base_id"
            ].astype(str)
            ==
            base_id
        )
        &
        (
            results[
                "prompt_id"
            ].astype(int)
            ==
            prompt_id
        )
    )


    idx = results.index[
        mask
    ][0]


    h = pre[idx]


    h_unit = (
        h
        /
        max(
            np.linalg.norm(h),
            1e-8
        )
    )


    similarities = (
        reference_unit
        @ h_unit
    )


    neighbor_idx = np.argpartition(
        -similarities,
        K_NEIGHBORS
    )[:K_NEIGHBORS]


    neighbors = (
        reference[
            neighbor_idx
        ]
    )


    center = (
        neighbors.mean(
            axis=0
        )
    )


    centered_neighbors = (
        neighbors
        - center
    )


    pca = PCA(
        n_components=
            VARIANCE_RETAINED,
        svd_solver="full"
    )


    pca.fit(
        centered_neighbors
    )


    key = (
        base_id,
        prompt_id
    )


    tangent_spaces[
        key
    ] = {
        "center":
            center,

        "basis":
            pca.components_,

        "n_components":
            int(
                pca.n_components_
            ),

        "mean_neighbor_cosine":
            float(
                similarities[
                    neighbor_idx
                ].mean()
            ),
    }


    print(
        f"[{n+1}/"
        f"{len(unique_prompts)}] "
        f"{base_id} "
        f"PCA dims="
        f"{pca.n_components_}"
    )


# ============================================================
# PER-RUN ALIGNMENT + GEOMETRY
# ============================================================

rows = []


for i, row in results.iterrows():

    base_id = str(
        row[
            "base_id"
        ]
    )

    prompt_id = int(
        row[
            "prompt_id"
        ]
    )

    vector_id = int(
        row[
            "random_vector_id"
        ]
    )


    geom = tangent_spaces[
        (
            base_id,
            prompt_id
        )
    ]


    basis = geom[
        "basis"
    ]

    center = geom[
        "center"
    ]


    delta = (
        actual_intervention[
            i
        ]
    )


    # --------------------------------------------------------
    # Tangent / normal decomposition
    # --------------------------------------------------------

    delta_tangent = (
        basis.T
        @ (
            basis
            @ delta
        )
    )


    delta_normal = (
        delta
        -
        delta_tangent
    )


    total_norm = float(
        np.linalg.norm(
            delta
        )
    )


    tangent_norm = float(
        np.linalg.norm(
            delta_tangent
        )
    )


    normal_norm = float(
        np.linalg.norm(
            delta_normal
        )
    )


    normal_fraction = (
        normal_norm
        /
        total_norm
        if total_norm > 1e-8
        else 0.0
    )


    normal_energy_fraction = (
        normal_norm ** 2
        /
        total_norm ** 2
        if total_norm > 1e-8
        else 0.0
    )


    # --------------------------------------------------------
    # Local-PCA residual before / after steering
    # --------------------------------------------------------

    h_pre = pre[i]

    h_post = post[i]


    pre_centered = (
        h_pre
        -
        center
    )


    post_centered = (
        h_post
        -
        center
    )


    pre_proj = (
        basis.T
        @ (
            basis
            @ pre_centered
        )
    )


    post_proj = (
        basis.T
        @ (
            basis
            @ post_centered
        )
    )


    pre_residual = (
        pre_centered
        -
        pre_proj
    )


    post_residual = (
        post_centered
        -
        post_proj
    )


    pre_residual_norm = float(
        np.linalg.norm(
            pre_residual
        )
    )


    post_residual_norm = float(
        np.linalg.norm(
            post_residual
        )
    )


    delta_residual = (
        post_residual_norm
        -
        pre_residual_norm
    )


    # --------------------------------------------------------
    # Movement along Qwen refusal-associated direction
    # --------------------------------------------------------

    actual_projection_minus_r = float(
        np.dot(
            delta,
            minus_r
        )
    )


    rows.append(
        {
            "base_id":
                base_id,

            "prompt_id":
                prompt_id,

            "random_vector_id":
                vector_id,

            "coefficient_c":
                float(
                    row[
                        "coefficient_c"
                    ]
                ),

            "alpha":
                float(
                    row[
                        "alpha"
                    ]
                ),

            "rough_refusal":
                bool(
                    row[
                        "rough_refusal"
                    ]
                ),

            "rough_nonrefusal":
                bool(
                    row[
                        "rough_nonrefusal"
                    ]
                ),

            "degenerate_flag":
                bool(
                    row[
                        "degenerate_flag"
                    ]
                ),

            "screening_label":
                str(
                    row[
                        "screening_label"
                    ]
                ),

            "cosine_with_minus_r":
                alignment[
                    vector_id
                ],

            "actual_projection_minus_r":
                actual_projection_minus_r,

            "total_norm":
                total_norm,

            "tangent_norm":
                tangent_norm,

            "normal_norm":
                normal_norm,

            "normal_norm_fraction":
                normal_fraction,

            "normal_energy_fraction":
                normal_energy_fraction,

            "pre_residual_norm":
                pre_residual_norm,

            "post_residual_norm":
                post_residual_norm,

            "delta_residual_norm":
                delta_residual,

            "local_pca_components":
                geom[
                    "n_components"
                ],

            "mean_neighbor_cosine":
                geom[
                    "mean_neighbor_cosine"
                ],
        }
    )


per_run = pd.DataFrame(
    rows
)


per_run.to_csv(
    OUT_DIR
    / "qwen_alignment_geometry_per_run.csv",
    index=False
)


# ============================================================
# PRIMARY CONDITION: c = 0.75
# ============================================================

primary = per_run[
    np.isclose(
        per_run[
            "coefficient_c"
        ],
        PRIMARY_COEFFICIENT
    )
].copy()


print("\n")
print("=" * 120)
print(
    "PRIMARY QWEN ANALYSIS: c = 0.75"
)
print("=" * 120)


print(
    "Runs:",
    len(primary)
)

print(
    "Refusals:",
    int(
        primary[
            "rough_refusal"
        ].sum()
    )
)

print(
    "Degenerate:",
    int(
        primary[
            "degenerate_flag"
        ].sum()
    )
)

print(
    "Coherent non-refusal candidates:",
    int(
        (
            primary[
                "screening_label"
            ]
            ==
            "coherent_nonrefusal_candidate"
        ).sum()
    )
)


# ============================================================
# VECTOR-LEVEL SUMMARY
# ============================================================

vector_summary = (
    primary
    .groupby(
        "random_vector_id",
        as_index=False
    )
    .agg(
        n=(
            "base_id",
            "size"
        ),

        refusal_rate=(
            "rough_refusal",
            "mean"
        ),

        degeneration_rate=(
            "degenerate_flag",
            "mean"
        ),

        cosine_with_minus_r=(
            "cosine_with_minus_r",
            "first"
        ),

        mean_projection_minus_r=(
            "actual_projection_minus_r",
            "mean"
        ),

        mean_tangent_norm=(
            "tangent_norm",
            "mean"
        ),

        mean_normal_norm=(
            "normal_norm",
            "mean"
        ),

        mean_normal_fraction=(
            "normal_norm_fraction",
            "mean"
        ),

        mean_delta_residual=(
            "delta_residual_norm",
            "mean"
        ),
    )
)


candidate_counts = (
    primary[
        primary[
            "screening_label"
        ]
        ==
        "coherent_nonrefusal_candidate"
    ]
    .groupby(
        "random_vector_id"
    )
    .size()
)


vector_summary[
    "coherent_nonrefusal_count"
] = (
    vector_summary[
        "random_vector_id"
    ]
    .map(
        candidate_counts
    )
    .fillna(0)
    .astype(int)
)


vector_summary[
    "coherent_nonrefusal_rate"
] = (
    vector_summary[
        "coherent_nonrefusal_count"
    ]
    /
    vector_summary[
        "n"
    ]
)


vector_summary.to_csv(
    OUT_DIR
    / "qwen_c075_vector_summary.csv",
    index=False
)


print("\n")
print("=" * 150)
print("QWEN c=0.75 VECTOR SUMMARY")
print("=" * 150)

print(
    vector_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.5f}"
    )
)


# ============================================================
# VECTOR-LEVEL CORRELATIONS
# ============================================================

metrics = [
    "cosine_with_minus_r",
    "mean_projection_minus_r",
    "mean_tangent_norm",
    "mean_normal_norm",
    "mean_normal_fraction",
    "mean_delta_residual",
]


corr_rows = []


print("\n")
print("=" * 120)
print(
    "VECTOR-LEVEL CORRELATIONS WITH "
    "COHERENT NON-REFUSAL RATE"
)
print("=" * 120)


for metric in metrics:

    rho, p = spearmanr(
        vector_summary[
            metric
        ],
        vector_summary[
            "coherent_nonrefusal_rate"
        ]
    )


    corr_rows.append(
        {
            "metric":
                metric,

            "spearman_rho":
                float(
                    rho
                ),

            "p_value":
                float(
                    p
                ),
        }
    )


    print(
        f"{metric:32s} "
        f"rho={rho:+.4f} "
        f"p={p:.4f}"
    )


pd.DataFrame(
    corr_rows
).to_csv(
    OUT_DIR
    / "qwen_c075_vector_correlations.csv",
    index=False
)


# ============================================================
# CLEAN REFUSAL VS COHERENT NON-REFUSAL COMPARISON
# ============================================================

clean = primary[
    primary[
        "screening_label"
    ].isin(
        [
            "refusal",
            "coherent_nonrefusal_candidate",
        ]
    )
].copy()


refusal = clean[
    clean[
        "screening_label"
    ]
    ==
    "refusal"
]


candidate = clean[
    clean[
        "screening_label"
    ]
    ==
    "coherent_nonrefusal_candidate"
]


print("\n")
print("=" * 120)
print(
    "QWEN c=0.75 CLEAN RUN COMPARISON"
)
print("=" * 120)


print(
    "Refusal runs:",
    len(refusal)
)

print(
    "Coherent non-refusal candidates:",
    len(candidate)
)


comparison_metrics = [
    "actual_projection_minus_r",
    "tangent_norm",
    "normal_norm",
    "normal_norm_fraction",
    "pre_residual_norm",
    "post_residual_norm",
    "delta_residual_norm",
]


comparison_rows = []


for metric in comparison_metrics:

    refusal_median = (
        refusal[
            metric
        ].median()
    )

    candidate_median = (
        candidate[
            metric
        ].median()
    )


    comparison_rows.append(
        {
            "metric":
                metric,

            "refusal_median":
                refusal_median,

            "coherent_nonrefusal_median":
                candidate_median,
        }
    )


    print(
        f"\n{metric}"
    )

    print(
        "  refusal median:   ",
        f"{refusal_median:.5f}"
    )

    print(
        "  candidate median: ",
        f"{candidate_median:.5f}"
    )


pd.DataFrame(
    comparison_rows
).to_csv(
    OUT_DIR
    / "qwen_c075_clean_comparison.csv",
    index=False
)


# ============================================================
# ALL-STRENGTH SUMMARY
# ============================================================

all_summary = (
    per_run
    .groupby(
        "coefficient_c",
        as_index=False
    )
    .agg(
        n=(
            "base_id",
            "size"
        ),

        refusal_rate=(
            "rough_refusal",
            "mean"
        ),

        degeneration_rate=(
            "degenerate_flag",
            "mean"
        ),

        mean_projection_minus_r=(
            "actual_projection_minus_r",
            "mean"
        ),

        mean_normal_fraction=(
            "normal_norm_fraction",
            "mean"
        ),

        mean_delta_residual=(
            "delta_residual_norm",
            "mean"
        ),
    )
)


all_summary.to_csv(
    OUT_DIR
    / "qwen_all_strengths_geometry_summary.csv",
    index=False
)


print("\n")
print("=" * 120)
print(
    "QWEN ALL-STRENGTH GEOMETRY SUMMARY"
)
print("=" * 120)

print(
    all_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.5f}"
    )
)


print("\nSaved to:")
print(
    OUT_DIR
)

print(
    "\nNOTE: coherent non-refusal is still "
    "a screening label, not confirmed "
    "harmful compliance."
)

print(
    "Vector-level analysis has only "
    "10 random directions, so correlations "
    "are exploratory."
)
