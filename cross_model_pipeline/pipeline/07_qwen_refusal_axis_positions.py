from pathlib import Path
import numpy as np
import pandas as pd

from safetensors.torch import load_file
from scipy.stats import mannwhitneyu, spearmanr


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

DIRECTION_FILE = (
    MODEL_DIR
    / "matched_refusal_direction"
    / "matched_refusal_direction.safetensors"
)

OUT_DIR = (
    MODEL_DIR
    / "refusal_axis_analysis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PRIMARY_COEFFICIENT = 0.75


# ============================================================
# LOAD
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

assert len(results) == len(pre) == len(post)


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


# ============================================================
# REFUSAL-AXIS SCORES
#
# Higher score = more aligned with the refusal-associated side.
# ============================================================

pre_score = (
    pre
    @ r
)

post_score = (
    post
    @ r
)

delta_score = (
    post_score
    - pre_score
)


analysis = results.copy()

analysis[
    "refusal_score_pre"
] = pre_score

analysis[
    "refusal_score_post"
] = post_score

analysis[
    "delta_refusal_score"
] = delta_score


analysis.to_csv(
    OUT_DIR
    / "qwen_refusal_axis_per_run.csv",
    index=False,
)


# ============================================================
# PRIMARY CONDITION
# ============================================================

primary = analysis[
    np.isclose(
        analysis[
            "coefficient_c"
        ],
        PRIMARY_COEFFICIENT
    )
].copy()


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
print("=" * 100)
print(
    "QWEN REFUSAL-AXIS ANALYSIS: c = 0.75"
)
print("=" * 100)

print(
    "Refusal runs:",
    len(refusal)
)

print(
    "Coherent non-refusal candidates:",
    len(candidate)
)


metrics = [
    "refusal_score_pre",
    "refusal_score_post",
    "delta_refusal_score",
]


rows = []


for metric in metrics:

    r_med = (
        refusal[
            metric
        ].median()
    )

    c_med = (
        candidate[
            metric
        ].median()
    )

    r_mean = (
        refusal[
            metric
        ].mean()
    )

    c_mean = (
        candidate[
            metric
        ].mean()
    )


    try:

        u, p = mannwhitneyu(
            refusal[
                metric
            ],
            candidate[
                metric
            ],
            alternative="two-sided",
        )

    except Exception:

        u = np.nan
        p = np.nan


    rows.append(
        {
            "metric":
                metric,

            "refusal_median":
                r_med,

            "candidate_median":
                c_med,

            "refusal_mean":
                r_mean,

            "candidate_mean":
                c_mean,

            "mannwhitney_u":
                u,

            "p_value":
                p,
        }
    )


    print(
        f"\n{metric}"
    )

    print(
        "  refusal median:   ",
        f"{r_med:.6f}"
    )

    print(
        "  candidate median: ",
        f"{c_med:.6f}"
    )

    print(
        "  refusal mean:     ",
        f"{r_mean:.6f}"
    )

    print(
        "  candidate mean:   ",
        f"{c_mean:.6f}"
    )

    print(
        "  Mann-Whitney p:   ",
        f"{p:.6g}"
    )


pd.DataFrame(
    rows
).to_csv(
    OUT_DIR
    / "qwen_refusal_axis_clean_comparison.csv",
    index=False,
)


# ============================================================
# OPTIONAL: BINARY OUTCOME CORRELATION
#
# coherent non-refusal = 1
# refusal = 0
# ============================================================

clean = clean.copy()

clean[
    "nonrefusal_outcome"
] = (
    clean[
        "screening_label"
    ]
    ==
    "coherent_nonrefusal_candidate"
).astype(int)


print("\n")
print("=" * 100)
print(
    "SPEARMAN WITH COHERENT NON-REFUSAL OUTCOME"
)
print("=" * 100)


corr_rows = []


for metric in metrics:

    rho, p = spearmanr(
        clean[
            metric
        ],
        clean[
            "nonrefusal_outcome"
        ],
    )

    corr_rows.append(
        {
            "metric":
                metric,

            "spearman_rho":
                rho,

            "p_value":
                p,
        }
    )


    print(
        f"{metric:25s} "
        f"rho={rho:+.4f} "
        f"p={p:.6g}"
    )


pd.DataFrame(
    corr_rows
).to_csv(
    OUT_DIR
    / "qwen_refusal_axis_correlations.csv",
    index=False,
)


# ============================================================
# ALL STRENGTHS SUMMARY
# ============================================================

all_strengths = (
    analysis
    .groupby(
        "coefficient_c",
        as_index=False,
    )
    .agg(
        n=(
            "base_id",
            "size",
        ),

        refusal_rate=(
            "rough_refusal",
            "mean",
        ),

        mean_pre_score=(
            "refusal_score_pre",
            "mean",
        ),

        mean_post_score=(
            "refusal_score_post",
            "mean",
        ),

        mean_delta_score=(
            "delta_refusal_score",
            "mean",
        ),
    )
)


print("\n")
print("=" * 100)
print(
    "ALL-STRENGTH REFUSAL-AXIS SUMMARY"
)
print("=" * 100)

print(
    all_strengths.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.5f}"
    )
)


all_strengths.to_csv(
    OUT_DIR
    / "qwen_all_strength_refusal_axis_summary.csv",
    index=False,
)


print("\nSaved to:")
print(
    OUT_DIR
)
