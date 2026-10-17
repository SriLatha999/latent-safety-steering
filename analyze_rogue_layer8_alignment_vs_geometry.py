import os
import glob
import numpy as np
import pandas as pd
import torch

from safetensors.torch import load_file
from sklearn.decomposition import PCA
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = "/storage/ice1/3/9/sbondala3/olmo_ood"

RESULTS_FILE = os.path.join(
    PROJECT_DIR,
    "rogue_scalpel_layer8_focused",
    "rogue_layer8_focused_results.csv",
)

ACT_FILE = os.path.join(
    PROJECT_DIR,
    "rogue_scalpel_layer8_focused",
    "rogue_layer8_focused_activations.safetensors",
)

RANDOM_VECTOR_FILE = os.path.join(
    PROJECT_DIR,
    "rogue_scalpel_layer8_focused",
    "random_vectors.safetensors",
)

REFUSAL_DIRECTION_FILE = os.path.join(
    PROJECT_DIR,
    "rogue_scalpel_layer8_focused",
    "refusal_direction_layer8",
    "refusal_direction_block8.safetensors",
)

REFERENCE_DIR = os.path.join(
    PROJECT_DIR,
    "current_pipeline_reference_layer8",
)

OUT_DIR = os.path.join(
    PROJECT_DIR,
    "rogue_scalpel_layer8_focused",
    "alignment_vs_geometry",
)

os.makedirs(OUT_DIR, exist_ok=True)

K_NEIGHBORS = 100
VARIANCE_RETAINED = 0.90

PRIMARY_COEFFICIENT = 1.0


# ============================================================
# LOAD BEHAVIOR + ACTIVATIONS
# ============================================================

results = pd.read_csv(
    RESULTS_FILE
)

acts = load_file(
    ACT_FILE
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
    == len(pre)
    == len(post)
    == len(actual_intervention)
)

print("Runs:", len(results))
print("Pre shape:", pre.shape)


# ============================================================
# LOAD RANDOM VECTORS
# ============================================================

rv = load_file(
    RANDOM_VECTOR_FILE
)

random_vectors = {}

for key, tensor in rv.items():

    if key.startswith(
        "random_vector_"
    ):

        vector_id = int(
            key.split("_")[-1]
        )

        v = tensor.float().numpy()

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
    len(random_vectors),
)


# ============================================================
# LOAD LAYER-8 REFUSAL DIRECTION
# ============================================================

dirs = load_file(
    REFUSAL_DIRECTION_FILE
)

r = (
    dirs[
        "refusal_direction_block_8"
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
    "Refusal direction norm:",
    np.linalg.norm(r),
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
            minus_r,
        )
    )


print("\nRandom-vector alignment with -r:")

for vector_id in sorted(
    alignment
):

    print(
        f"vector {vector_id:03d}: "
        f"{alignment[vector_id]:+.6f}"
    )


# ============================================================
# LOAD FULL LAYER-8 REFERENCE BANK
# ============================================================

print(
    "\nLoading layer-8 reference shards..."
)

shard_files = sorted(
    glob.glob(
        os.path.join(
            REFERENCE_DIR,
            "activations_*.safetensors",
        )
    )
)

if not shard_files:

    raise RuntimeError(
        "No reference activation shards found."
    )


reference_parts = []

for i, path in enumerate(
    shard_files,
    start=1,
):

    shard = load_file(
        path
    )

    x = (
        shard[
            "prompt_final_layer8"
        ]
        .float()
        .numpy()
    )

    reference_parts.append(
        x
    )

    print(
        f"[{i}/{len(shard_files)}] "
        f"{os.path.basename(path)} "
        f"{x.shape}"
    )


reference = np.concatenate(
    reference_parts,
    axis=0,
)

print(
    "\nReference shape:",
    reference.shape,
)


# ============================================================
# COSINE-SEARCH VERSION OF REFERENCE BANK
# ============================================================

reference_norm = np.linalg.norm(
    reference,
    axis=1,
    keepdims=True,
)

reference_unit = (
    reference
    /
    np.clip(
        reference_norm,
        1e-8,
        None,
    )
)


# ============================================================
# FIT ONE LOCAL PCA PER TEST PROMPT
#
# Geometry should be estimated from the UNSTEERED
# layer-8 prompt activation.
#
# Because each prompt appears many times across vectors/c,
# fit its local tangent only once.
# ============================================================

unique_attack_ids = (
    results["attack_id"]
    .astype(str)
    .unique()
)

print(
    "\nUnique test prompts:",
    len(unique_attack_ids),
)


tangent_spaces = {}


for n, attack_id in enumerate(
    unique_attack_ids,
    start=1,
):

    idx = results.index[
        results["attack_id"]
        .astype(str)
        == attack_id
    ][0]

    h = pre[idx]

    h_unit = (
        h
        /
        max(
            np.linalg.norm(h),
            1e-8,
        )
    )

    similarities = (
        reference_unit
        @ h_unit
    )

    neighbor_idx = np.argpartition(
        -similarities,
        K_NEIGHBORS,
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
        svd_solver="full",
    )

    pca.fit(
        centered_neighbors
    )

    tangent_spaces[
        attack_id
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
        f"[{n}/"
        f"{len(unique_attack_ids)}] "
        f"{attack_id} "
        f"PCA dims="
        f"{pca.n_components_}"
    )


# ============================================================
# PER-RUN ALIGNMENT + GEOMETRY
# ============================================================

rows = []


for i, row in results.iterrows():

    attack_id = str(
        row["attack_id"]
    )

    vector_id = int(
        row[
            "random_vector_id"
        ]
    )

    geom = tangent_spaces[
        attack_id
    ]

    basis = geom["basis"]
    center = geom["center"]

    delta = (
        actual_intervention[i]
    )

    # --------------------------------------------------------
    # TANGENT / NORMAL DECOMPOSITION
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
        - delta_tangent
    )

    total_norm = float(
        np.linalg.norm(delta)
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

    normal_norm_fraction = (
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
    # ENDPOINT LOCAL-PCA RESIDUAL
    # --------------------------------------------------------

    h_pre = pre[i]
    h_post = post[i]

    pre_centered = (
        h_pre
        - center
    )

    post_centered = (
        h_post
        - center
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
        - pre_proj
    )

    post_residual = (
        post_centered
        - post_proj
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
    # REFUSAL DIRECTION PROJECTION
    # --------------------------------------------------------

    actual_minus_r_projection = float(
        np.dot(
            delta,
            minus_r,
        )
    )


    rows.append(
        {
            "attack_id":
                attack_id,

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
                    row["alpha"]
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

            "cosine_with_minus_r8":
                alignment[
                    vector_id
                ],

            "actual_projection_minus_r8":
                actual_minus_r_projection,

            "total_norm":
                total_norm,

            "tangent_norm":
                tangent_norm,

            "normal_norm":
                normal_norm,

            "normal_norm_fraction":
                normal_norm_fraction,

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
    os.path.join(
        OUT_DIR,
        "layer8_alignment_geometry_per_run.csv",
    ),
    index=False,
)


# ============================================================
# PRIMARY ANALYSIS: c = 1.0
# ============================================================

primary = per_run[
    np.isclose(
        per_run[
            "coefficient_c"
        ],
        PRIMARY_COEFFICIENT,
    )
].copy()


print("\n")
print("=" * 120)
print(
    "PRIMARY ANALYSIS: "
    "ROGUE SCALPEL c = 1.0"
)
print("=" * 120)

print(
    "Runs:",
    len(primary),
)

print(
    "Refusal:",
    int(
        primary[
            "rough_refusal"
        ].sum()
    ),
)

print(
    "Degenerate:",
    int(
        primary[
            "degenerate_flag"
        ].sum()
    ),
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
    ),
)


# ============================================================
# VECTOR-LEVEL SUMMARY AT c=1.0
# ============================================================

vector_summary = (
    primary
    .groupby(
        "random_vector_id",
        as_index=False,
    )
    .agg(
        n=(
            "attack_id",
            "size",
        ),

        refusal_rate=(
            "rough_refusal",
            "mean",
        ),

        rough_nonrefusal_rate=(
            "rough_nonrefusal",
            "mean",
        ),

        degeneration_rate=(
            "degenerate_flag",
            "mean",
        ),

        cosine_with_minus_r8=(
            "cosine_with_minus_r8",
            "first",
        ),

        mean_projection_minus_r8=(
            "actual_projection_minus_r8",
            "mean",
        ),

        mean_tangent_norm=(
            "tangent_norm",
            "mean",
        ),

        mean_normal_norm=(
            "normal_norm",
            "mean",
        ),

        mean_normal_norm_fraction=(
            "normal_norm_fraction",
            "mean",
        ),

        mean_normal_energy_fraction=(
            "normal_energy_fraction",
            "mean",
        ),

        mean_delta_residual=(
            "delta_residual_norm",
            "mean",
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
    vector_summary["n"]
)


vector_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "c1_vector_summary.csv",
    ),
    index=False,
)


print("\n")
print("=" * 150)
print(
    "c=1.0 RANDOM VECTOR SUMMARY"
)
print("=" * 150)

print(
    vector_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.5f}",
    )
)


# ============================================================
# CORRELATIONS WITH COHERENT NON-REFUSAL RATE
# ============================================================

metrics = [
    "cosine_with_minus_r8",
    "mean_projection_minus_r8",
    "mean_tangent_norm",
    "mean_normal_norm",
    "mean_normal_norm_fraction",
    "mean_normal_energy_fraction",
    "mean_delta_residual",
]


corr_rows = []


print("\n")
print("=" * 110)
print(
    "VECTOR-LEVEL CORRELATIONS "
    "WITH COHERENT NON-REFUSAL RATE"
)
print("=" * 110)


for metric in metrics:

    rho, p = spearmanr(
        vector_summary[
            metric
        ],
        vector_summary[
            "coherent_nonrefusal_rate"
        ],
    )

    corr_rows.append(
        {
            "metric":
                metric,

            "spearman_rho":
                float(rho),

            "p_value":
                float(p),
        }
    )

    print(
        f"{metric:34s} "
        f"rho={rho:+.4f} "
        f"p={p:.4f}"
    )


pd.DataFrame(
    corr_rows
).to_csv(
    os.path.join(
        OUT_DIR,
        "c1_vector_correlations.csv",
    ),
    index=False,
)


# ============================================================
# PER-RUN GROUP COMPARISON
#
# Refusal vs coherent non-refusal candidate.
# Exclude obvious degenerate outputs.
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
print("=" * 110)
print(
    "c=1.0 CLEAN RUN COMPARISON"
)
print("=" * 110)

print(
    "Refusal runs:",
    len(refusal),
)

print(
    "Coherent non-refusal candidates:",
    len(candidate),
)


for metric in [
    "actual_projection_minus_r8",
    "tangent_norm",
    "normal_norm",
    "normal_norm_fraction",
    "pre_residual_norm",
    "post_residual_norm",
    "delta_residual_norm",
]:

    print(
        f"\n{metric}"
    )

    print(
        "  refusal median:    ",
        f"{refusal[metric].median():.5f}",
    )

    print(
        "  candidate median:  ",
        f"{candidate[metric].median():.5f}",
    )


# ============================================================
# SECONDARY SUMMARIES FOR ALL COEFFICIENTS
# ============================================================

all_summary = (
    per_run
    .groupby(
        "coefficient_c",
        as_index=False,
    )
    .agg(
        n=(
            "attack_id",
            "size",
        ),

        refusal_rate=(
            "rough_refusal",
            "mean",
        ),

        degeneration_rate=(
            "degenerate_flag",
            "mean",
        ),

        mean_projection_minus_r8=(
            "actual_projection_minus_r8",
            "mean",
        ),

        mean_normal_fraction=(
            "normal_norm_fraction",
            "mean",
        ),

        mean_delta_residual=(
            "delta_residual_norm",
            "mean",
        ),
    )
)


all_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "all_coefficients_geometry_summary.csv",
    ),
    index=False,
)


print("\n")
print("=" * 110)
print("ALL COEFFICIENTS GEOMETRY SUMMARY")
print("=" * 110)

print(
    all_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.5f}",
    )
)


print("\nSaved to:")
print(OUT_DIR)

print(
    "\nNOTE:"
)

print(
    "The coherent-nonrefusal label is still "
    "a screening label, not a harmful-compliance "
    "judgment."
)

print(
    "Vector-level n=10, so correlations are "
    "exploratory."
)
