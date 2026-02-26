"""
main.py – Inference demo and full-pipeline reference script.

Active code:
    Runs a single text-generation query against a saved checkpoint.

Commented pipeline (uncomment steps as needed):
    1. create_config_folders() – initialise workspace
    2. ApiCardSetup.set()      – register model + dataset names
    3. DataLoader.run()        – download from HuggingFace Hub
    4. Manager.dataset_prepare() – tokenise and format datasets
    5. FinetuneModel.finetune_model() – fine-tune the model

TODO:
    1. Accept only pre-formatted datasets (user provides acceptable format)
    2. Switch inference and training to the unsloth library
    3. Add push-to-HuggingFace-Hub after training
    4. Rewrite model architecture using Keras only
    5. Use unsloth GPT-format for dataset formatting
    6. Expose configurable model parameters
    7. Migrate config files from JSON to INI format
"""

from pathlib import Path

from modules.inference import InferenceManager

# ── Full pipeline (uncomment to run each step) ─────────────────────────────────

# from modules.variable import Variable
# from modules.prerun import create_config_folders
# from modules.ApiDump import ApiCardSetup
# from modules.DataDownload import DataLoader
# from modules.DataModelPrepare import Manager
# from modules.train import FinetuneModel

# variable = Variable()
# api      = variable.hf_api

# Step 1: initialise workspace config folders
# create_config_folders()

# Step 2: register models and datasets in the API card
# setcard     = ApiCardSetup()
# list_models = api.list_models(model_name='HuggingFaceTB/SmolLM2-360M', limit=1, gated=False)
# list_datasets = api.list_datasets(dataset_name='Lin-Chen/ShareGPT4V', limit=1, gated=False)
# list_download = setcard.set(list_models, list_datasets)

# Step 3: download models and datasets from the Hub
# DataLoader().run(list_download)

# Step 4: format datasets for training
# Manager().dataset_prepare(list_download)

# Step 5: fine-tune the model on the formatted datasets
# FinetuneModel().finetune_model()

# ── Inference demo ─────────────────────────────────────────────────────────────

model_path = (
    Path(__file__).parent
    / "checkpoints"
    / "text-generation"
    / "HuggingFaceTB_SmolLM2-360M"
)

inference_manager = InferenceManager(str(model_path))

user_input = "What is the total value in the image?"
response   = inference_manager.generate_response(user_input)
print(response)
