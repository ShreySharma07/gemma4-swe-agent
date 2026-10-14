"""
Kaggle accounts used by the notebook builders (make_eval_notebook.py, make_train_notebook.py).

KAGGLE_OWNER      account that PUSHES and runs the generated notebooks (default: shreysharma07). A teammate sets
                  their own username here; their notebook ids become <them>/check-l4, <them>/gemma4-train-lora, ...
KAGGLE_SRC_OWNER  account that owns the shared INPUTS: kernel outputs gemma4-scorer-env, gemma4-train-wheels,
                  gemma4-devset-build and the dataset test-deps-wheels (default: shreysharma07). The owner shares them
                  with the teammate on Kaggle (Share -> add collaborator), so nothing has to be rebuilt.
"""
import os

OWNER = os.environ.get("KAGGLE_OWNER", "shreysharma07")
SRC_OWNER = os.environ.get("KAGGLE_SRC_OWNER", "shreysharma07")


def mine(slug):
    return f"{OWNER}/{slug}"


def src(slug):
    return f"{SRC_OWNER}/{slug}"
