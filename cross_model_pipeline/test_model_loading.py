import argparse
import os
import sys

ROOT = "/storage/ice1/3/9/sbondala3/olmo_ood"

sys.path.append(
    os.path.join(
        ROOT,
        "cross_model_pipeline",
        "pipeline",
    )
)

from common import load_model


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    required=True,
)

args = parser.parse_args()


bundle = load_model(
    args.model
)

print("\nSUCCESS")

print(
    "Layer index:",
    bundle["layer_index"]
)

print(
    "Hidden size:",
    bundle["hidden_size"]
)

