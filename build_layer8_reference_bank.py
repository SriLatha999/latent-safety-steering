import os
import json
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import save_file


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "allenai/OLMo-2-0425-1B-SFT"

PROJECT_DIR = "/storage/ice1/3/9/sbondala3/olmo_ood"

# Use the SAME selected reference prompts used for the
# current-pipeline layer-16 reference bank.
REFERENCE_METADATA = os.path.join(
    PROJECT_DIR,
    "current_pipeline_reference_full",
    "reference_metadata_selected.csv",
)

OUT_DIR = os.path.join(
    PROJECT_DIR,
    "current_pipeline_reference_layer8",
)

os.makedirs(OUT_DIR, exist_ok=True)

TARGET_LAYER = 8

MAX_INPUT_TOKENS = 2048

# Save one shard every N prompts.
SHARD_SIZE = 500


# ============================================================
# HELPERS
# ============================================================

def format_prompt(tokenizer, prompt):

    messages = [
        {
            "role": "user",
            "content": str(prompt),
        }
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    except Exception:
        return str(prompt)


def get_hidden(output):

    if isinstance(output, tuple):
        return output[0]

    return output


# ============================================================
# LOAD REFERENCE METADATA
# ============================================================

df = pd.read_csv(
    REFERENCE_METADATA
)

print("Metadata rows:", len(df))
print("Columns:", list(df.columns))


# Try common prompt-column names.
prompt_candidates = [
    "prompt",
    "text",
    "instruction",
    "user_prompt",
    "attack_prompt",
    "input",
]

PROMPT_COL = None

for col in prompt_candidates:

    if col in df.columns:

        PROMPT_COL = col
        break


if PROMPT_COL is None:

    raise RuntimeError(
        "Could not identify prompt column.\n"
        f"Available columns: {list(df.columns)}"
    )


print("Using prompt column:", PROMPT_COL)


# ============================================================
# DETERMINE ALREADY-COMPLETED SHARDS
# ============================================================

completed_indices = set()

for filename in os.listdir(
    OUT_DIR
):

    if (
        filename.startswith(
            "metadata_"
        )
        and filename.endswith(
            ".csv"
        )
    ):

        shard_df = pd.read_csv(
            os.path.join(
                OUT_DIR,
                filename,
            )
        )

        completed_indices.update(
            shard_df[
                "reference_index"
            ].astype(int).tolist()
        )


print(
    "Already completed:",
    len(completed_indices),
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

device = next(
    model.parameters()
).device

hidden_size = (
    model.config.hidden_size
)

target_layer = (
    model.model.layers[
        TARGET_LAYER
    ]
)


print("Device:", device)
print("Hidden size:", hidden_size)
print(
    "Target:",
    f"model.layers[{TARGET_LAYER}]",
    type(target_layer).__name__,
)


# ============================================================
# HOOK
# ============================================================

hook_state = {
    "hidden": None,
}


def layer_hook(
    module,
    inputs,
    output,
):

    hidden = get_hidden(
        output
    )

    hook_state[
        "hidden"
    ] = (
        hidden.detach()
        .float()
        .cpu()
    )

    return output


handle = (
    target_layer.register_forward_hook(
        layer_hook
    )
)


# ============================================================
# SHARD BUFFERS
# ============================================================

activation_buffer = []
metadata_buffer = []

shard_number = 0


# Existing shard count so resumed runs don't overwrite.
existing_act_shards = [
    x for x in os.listdir(OUT_DIR)
    if x.startswith("activations_")
    and x.endswith(".safetensors")
]

if existing_act_shards:

    shard_numbers = []

    for name in existing_act_shards:

        try:
            n = int(
                name
                .replace(
                    "activations_",
                    ""
                )
                .replace(
                    ".safetensors",
                    ""
                )
            )

            shard_numbers.append(n)

        except Exception:
            pass

    if shard_numbers:
        shard_number = max(
            shard_numbers
        ) + 1


def save_shard():

    global shard_number
    global activation_buffer
    global metadata_buffer

    if len(
        activation_buffer
    ) == 0:
        return

    tensor = torch.stack(
        activation_buffer,
        dim=0,
    )

    act_path = os.path.join(
        OUT_DIR,
        f"activations_{shard_number:04d}.safetensors",
    )

    meta_path = os.path.join(
        OUT_DIR,
        f"metadata_{shard_number:04d}.csv",
    )


    save_file(
        {
            "prompt_final_layer8":
                tensor
        },
        act_path,
    )

    pd.DataFrame(
        metadata_buffer
    ).to_csv(
        meta_path,
        index=False,
    )


    print(
        f"\nSAVED SHARD "
        f"{shard_number:04d}: "
        f"{len(activation_buffer)} rows"
    )

    print(
        "Shape:",
        tuple(tensor.shape),
    )


    shard_number += 1

    activation_buffer = []
    metadata_buffer = []


# ============================================================
# RUN REFERENCE PROMPTS
# ============================================================

usable = 0
skipped_existing = 0
failed = 0


with torch.inference_mode():

    for idx, row in df.iterrows():

        reference_index = int(idx)


        if (
            reference_index
            in completed_indices
        ):

            skipped_existing += 1
            continue


        prompt = str(
            row[PROMPT_COL]
        )


        formatted = format_prompt(
            tokenizer,
            prompt,
        )


        enc = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )


        input_ids = (
            enc["input_ids"]
            .to(device)
        )

        attention_mask = (
            enc["attention_mask"]
            .to(device)
        )


        hook_state[
            "hidden"
        ] = None


        try:

            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        except Exception as e:

            failed += 1

            print(
                "\nFAILED:",
                reference_index,
                str(e),
            )

            continue


        hidden = hook_state[
            "hidden"
        ]


        if hidden is None:

            failed += 1

            print(
                "\nHOOK FAILED:",
                reference_index,
            )

            continue


        # Since each prompt is processed independently,
        # last position is the final prompt token.
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


        # Preserve useful IDs from original metadata.
        for col in df.columns:

            if col == PROMPT_COL:
                continue

            value = row[col]

            if pd.isna(value):
                value = None

            meta[col] = value


        metadata_buffer.append(
            meta
        )


        usable += 1


        if (
            usable % 100
            == 0
        ):

            print(
                f"Processed new: "
                f"{usable} | "
                f"existing: "
                f"{skipped_existing} | "
                f"failed: {failed}"
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
print("=" * 80)
print("LAYER-8 REFERENCE EXTRACTION COMPLETE")
print("=" * 80)

print(
    "New usable:",
    usable,
)

print(
    "Already existed:",
    skipped_existing,
)

print(
    "Failed:",
    failed,
)

print(
    "Output:",
    OUT_DIR,
)
