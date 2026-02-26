from huggingface_hub import HfApi
from pathlib import Path
import torch


class Variable:
    """
    Central configuration object.
    All paths and settings for the project are derived from WORKSPACE
    (the repository root), so this is the single source of truth.
    """

    def __init__(self):
        # ── Workspace root ─────────────────────────────────────────────
        # Resolved to the repository root regardless of where Python is run from.
        self.WORKSPACE = Path(__file__).parent.parent.absolute()

        # ── HuggingFace ────────────────────────────────────────────────
        self.hf_api = HfApi()
        self.model_data_logs = {}

        # ── File / folder name constants ────────────────────────────────
        self.Api_card_file             = "ApiCardSet.json"
        self.configs_folder            = "configs"
        self.saved_configs_file        = "saved_config.json"
        self.repositories_folder       = "repositories"
        self.repositories_dataset_folder = "datasets"
        self.repositories_model_folder = "models"
        self.custom_model_folder       = "custom_models"
        self.vision_model_folder       = "vision-model"
        self.conversation_model_folder = "conversation-model"
        self.model_saved_folder        = "model-trained"
        self.model_checkpoints         = "checkpoints"

        # ── Config paths ───────────────────────────────────────────────
        self.DMConfig_DIR      = self.WORKSPACE / self.configs_folder
        self.Card_Path         = self.DMConfig_DIR / self.Api_card_file
        self.SAVED_CONFIG_Path = self.DMConfig_DIR / self.saved_configs_file

        # ── Repository paths (downloaded models / datasets) ────────────
        self.REPO_DIR      = self.WORKSPACE / self.repositories_folder
        self.DATASETS_DIR  = self.REPO_DIR  / self.repositories_dataset_folder
        self.LocalModel_DIR = self.REPO_DIR / self.repositories_model_folder

        # ── Custom model paths (wrapped / fine-tuned) ──────────────────
        self.CUSTOM_MODEL_DIR  = self.WORKSPACE / self.custom_model_folder
        self.VISION_MODEL_DIR  = self.CUSTOM_MODEL_DIR / self.vision_model_folder
        self.REGULAR_MODEL_DIR = self.CUSTOM_MODEL_DIR / self.conversation_model_folder

        # ── Training output paths ──────────────────────────────────────
        self.MODEL_DIR      = self.WORKSPACE / self.model_saved_folder
        self.CHECKPOINT_DIR = self.WORKSPACE / self.model_checkpoints
        self.OFFLOAD_DIR    = self.WORKSPACE / "offload"

        # Path where training_config.json is written after dataset_prepare()
        self.training_config_path = self.CUSTOM_MODEL_DIR / "training_config.json"

        # Path where tokenized / formatted datasets are saved
        self.DATASET_FORMATTED_DIR = self.WORKSPACE / "formatted_datasets"

        # ── Chat template ──────────────────────────────────────────────
        # Contains .jinja files that define how messages are rendered
        self.chat_template_path = self.WORKSPACE / "chat_template"

        # ── Torch dtype used across the project ────────────────────────
        self.DTYPE = torch.float32
