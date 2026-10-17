from pathlib import Path
import os
import sys

import pandas as pd
import torch
from safetensors.torch import save_file


# ============================================================
# PATHS / IMPORTS
# ============================================================

ROOT = Path(
    "/storage/ice1/3/9/sbondala3/olmo_ood"
)

sys.path.append(
    str(
        ROOT
        / "cross_model_pipeline"
        / "pipeline"
    )
)

from common import (
    load_model,
    format_prompt,
)


MODEL_KEY = "qwen"

REFERENCE_METADATA = (
    ROOT
    / "current_pipeline_reference_full"
    / "reference_metadata.csv"
)

OUT_DIR = (
    ROOT
    / "cross_model_results"
    / MODEL_KEY
    / "reference_bank"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

MAX_INPUT_TOKENS = 2048
SHARD_SIZE = 500


# ============================================================
# LOAD REFERENCE METADATA
# ============================================================

df = pd.read_csv(
    REFERENCE_METADATA
)

print(
    "Reference rows:",
    len(df)
)

print(
    "Columns:",
    list(df.columns)
)


prompt_candidates = [
    "prompt",
    "text",
    "instruction",
    "user_prompt",
    "input",
]


PROMPT_COL = None


for col in prompt_candidates:

    if col in df.columns:

        PROMPT_COL = col
        break


if PROMPT_COL is None:

    raise RuntimeError(
        "Could not identify prompt column."
    )


print(
    "Using prompt column:",
    PROMPT_COL
)


# ============================================================
# LOAD MODEL
# ============================================================

bundle = load_model(
    MODEL_KEY
)

model = bundle["model"]
tokenizer = bundle["tokenizer"]
layers = bundle["layers"]

TARGET_LAYER = bundle[
    "layer_index"
]

HIDDEN_SIZE = bundle[
    "hidden_size"
]

target_module = layers[
    TARGET_LAYER
]

device = next(
    model.parameters()
).device


print(
    "Target block:",
    TARGET_LAYER
)

print(
    "Hidden size:",
    HIDDEN_SIZE
)


# ============================================================
# RESUME SUPPORT
# ============================================================

completed_indices = set()


for path in sorted(
    OUT_DIR.glob(
        "metadata_*.csv"
    )
):

    shard_df = pd.read_csv(
        path
    )

    completed_indices.update(
        shard_df[
            "reference_index"
        ]
        .astype(int)
        .tolist()
    )


print(
    "Already completed:",
    len(
        completed_indices
    )
)


existing_shards = sorted(
    OUT_DIR.glob(
        "activations_*.safetensors"
    )
)


if existing_shards:

    shard_numbers = []

    for p in existing_shards:

        try:

            shard_numbers.append(
                int(
                    p.stem
                    .replace(
                        "activations_",
                        ""
                    )
                )
            )

        except Exception:
            pass


    next_shard = (
        max(
            shard_numbers
        )
        + 1
        if shard_numbers
        else 0
    )

else:

    next_shard = 0


# ============================================================
# HOOK
# ============================================================

state = {
    "hidden": None
}


def extract_hidden(output):

    if isinstance(
        output,
        tuple
    ):
        return output[0]

    if hasattr(
        output,
        "last_hidden_state"
    ):
        return output.last_hidden_state

    return output


def hook_fn(
    module,
    inputs,
    output,
):

    hidden = extract_hidden(
        output
    )

    state[
        "hidden"
    ] = (
        hidden
        .detach()
        .float()
        .cpu()
    )


handle = (
    target_module
    .register_forward_hook(
        hook_fn
    )
)


# ============================================================
# BUFFERS
# ============================================================

activation_buffer = []
metadata_buffer = []

shard_number = (
    next_shard
)


def save_shard():

    global activation_buffer
    global metadata_buffer
    global shard_number

    if len(
        activation_buffer
    ) == 0:
        return


    tensor = torch.stack(
        activation_buffer,
        dim=0,
    ).half()


    act_path = (
        OUT_DIR
        / f"activations_{shard_number:04d}.safetensors"
    )

    meta_path = (
        OUT_DIR
        / f"metadata_{shard_number:04d}.csv"
    )


    save_file(
        {
            "prompt_final":
                tensor
        },
        str(
            act_path
        ),
    )


    pd.DataFrame(
        metadata_buffer
    ).to_csv(
        meta_path,
        index=False,
    )


    print(
        f"\nSAVED SHARD "
        f"{shard_number:04d}"
    )

    print(
        "Rows:",
        len(
            activation_buffer
        )
    )

    print(
        "Shape:",
        tuple(
            tensor.shape
        )
    )


    shard_number += 1

    activation_buffer = []
    metadata_buffer = []


# ============================================================
# RUN
# ============================================================

new_usable = 0
failed = 0
skipped = 0


with torch.inference_mode():

    for idx, row in df.iterrows():

        reference_index = int(
            idx
        )


        if (
            reference_index
            in completed_indices
        ):

            skipped += 1
            continue


        prompt = str(
            row[
                PROMPT_COL
            ]
        )


        formatted = format_prompt(
            tokenizer,
            prompt,
        )


        enc = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=
                MAX_INPUT_TOKENS,
        )


        input_ids = (
            enc[
                "input_ids"
            ]
            .to(device)
        )

        attention_mask = (
            enc[
                "attention_mask"
            ]
            .to(device)
        )


        state[
            "hidden"
        ] = None


        try:

            _ = model(
                input_ids=input_ids,
                attention_mask=
                    attention_mask,
                use_cache=False,
                return_dict=True,
            )

        except Exception as e:

            failed += 1

            print(
                "\nFAILED",
                reference_index,
                str(e),
            )

            continue


        hidden = state[
            "hidden"
        ]


        if hidden is None:

            failed += 1

            print(
                "\nHOOK FAILED",
                reference_index,
            )

            continue


        activation = (
            hidden[
                0,
                -1,
                :
            ]
            .clone()
        )


        activation_buffer.append(
            activation
        )


        meta = {
            "reference_index":
                reference_index,

            "prompt":
                prompt,
        }


        for col in df.columns:

            if col == PROMPT_COL:
                continue

            value = row[
                col
            ]

            if pd.isna(
                value
            ):
                value = None

            meta[
                col
            ] = value


        metadata_buffer.append(
            meta
        )


        new_usable += 1


        if (
            new_usable % 100
            == 0
        ):

            print(
                f"Processed new: "
                f"{new_usable} | "
                f"existing: "
                f"{skipped} | "
                f"failed: "
                f"{failed}"
            )


        if (
            len(
                activation_buffer
            )
            >= SHARD_SIZE
        ):

            save_shard()


# ============================================================
# FINAL SHARD
# ============================================================

save_shard()

handle.remove()


print("\n")
print("=" * 100)
print("QWEN REFERENCE BANK COMPLETE")
print("=" * 100)

print(
    "New usable:",
    new_usable
)

print(
    "Already existed:",
    skipped
)

print(
    "Failed:",
    failed
)

print(
    "Output:",
    OUT_DIR
)
