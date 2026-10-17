import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = "/storage/ice1/3/9/sbondala3/olmo_ood"

sys.path.append(
    os.path.join(
        ROOT,
        "cross_model_pipeline",
        "configs",
    )
)

from models import MODELS


def load_model(model_key):

    if model_key not in MODELS:
        raise ValueError(
            f"Unknown model: {model_key}. "
            f"Choose from {list(MODELS.keys())}"
        )

    cfg = MODELS[model_key]

    model_name = cfg["model_name"]

    print("=" * 80)
    print("MODEL:", model_key)
    print("HF model:", model_name)
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    layers = get_transformer_layers(
        model,
        model_key,
    )

    n_layers = len(layers)

    relative_layer = cfg["relative_layer"]

    layer_index = round(
        relative_layer
        * (n_layers - 1)
    )

    hidden_size = model.config.hidden_size

    print("Number of layers:", n_layers)
    print("Hidden size:", hidden_size)
    print("Selected layer index:", layer_index)
    print(
        "Relative depth:",
        layer_index / (n_layers - 1)
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "layers": layers,
        "layer_index": layer_index,
        "hidden_size": hidden_size,
        "config": cfg,
    }


def get_transformer_layers(model, model_key):

    # Most decoder-only HuggingFace models
    if hasattr(model, "model"):

        if hasattr(model.model, "layers"):
            return model.model.layers

    # fallback
    if hasattr(model, "transformer"):

        if hasattr(model.transformer, "h"):
            return model.transformer.h

    raise RuntimeError(
        f"Could not identify transformer layers "
        f"for model {model_key}"
    )


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


def get_output_dir(model_key):

    path = os.path.join(
        ROOT,
        "cross_model_results",
        model_key,
    )

    os.makedirs(
        path,
        exist_ok=True,
    )

    return path
