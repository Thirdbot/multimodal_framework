"""
createmodel.py – Wrap a base model with LoRA and save it as a custom model.

Usage:
    python createmodel.py

    Reads the base model from local_models/<org>/<name>, applies LoRA
    adapters via CreateModel, and saves the result to custom_models/.

    Uncomment the vision block to create a multimodal vision model instead.
"""

from modules.variable import Variable
from modules.ModelUtils import CreateModel

# ── Paths ──────────────────────────────────────────────────────────────────────

vars       = Variable()
repo_folder = vars.LocalModel_DIR

# Base model to wrap (must be downloaded to local_models/ first)
oldmodel_path = repo_folder / "Qwen" / "Qwen1.5-0.5B-Chat"


# ── Conversation model ─────────────────────────────────────────────────────────
# Wraps the base model with ConversationModelWrapper + LoRA and saves it
# under custom_models/conversation-model/

create_conver_model = CreateModel(oldmodel_path, "conversation-model")
create_conver_model.add_conversation()
create_conver_model.save_regular_model()


# ── Vision model (optional) ────────────────────────────────────────────────────
# Wraps the base model with VisionModelWrapper + LoRA and saves it
# under custom_models/vision-model/

# create_vision_model = CreateModel(oldmodel_path, "vision-model")
# create_vision_model.add_vision()
# create_vision_model.save_vision_model()
