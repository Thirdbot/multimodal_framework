"""
DataDownload.py – Download models and datasets from HuggingFace Hub.

Works from both the GUI (via app.py handlers) and the command line.
"""

import json
from pathlib import Path

from colorama import Fore, Style, init
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from modules.variable import Variable

init(autoreset=True)

# File patterns that are downloaded for datasets
DATASET_ALLOWED_PATTERNS = ("*.json", "*.csv", "*.parquet", "*.zip")


# ── Dataset loader ─────────────────────────────────────────────────────────────

class FlexibleDatasetLoader:
    """
    Load and locally cache HuggingFace datasets.

    Remembers which config was used for each dataset so subsequent
    runs do not prompt again.
    """

    def __init__(self, split="train"):
        self.variable    = Variable()
        self.split       = split
        self.dataset     = None

        # Paths
        self.datasets_dir       = self.variable.DATASETS_DIR
        self.saved_config_file  = self.variable.SAVED_CONFIG_Path

        # Ensure target directory exists
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        # Load previously saved config choices from disk
        self.saved_configs = self._load_saved_configs()

    # ── Public ────────────────────────────────────────────────────────────────

    def load(self, name, config_name=None):
        """
        Load a dataset by HuggingFace repo ID.

        name        – repo ID, e.g. 'Lin-Chen/ShareGPT4V'
        config_name – dataset config subset name (auto-selected if None)
        """
        print(f"Dataset: {name}  Config: {config_name}")
        self.split = self._pick_split(name, config_name)

        if config_name:
            self._load_with_config(name, config_name)
        else:
            config_name = self._resolve_config(name)
            self.load(name, config_name)

    def get(self):
        """Return the loaded dataset object."""
        return self.dataset

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_with_config(self, name, config_name):
        """Attempt to load with the given config and download artifacts."""
        try:
            self.dataset = load_dataset(name, config_name, split=self.split)
            print(f"{Fore.GREEN}Loaded: {name}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{e}")
            # Retry without config
            self.load(name, None)
            return

        # Download raw files for local storage
        try:
            snapshot_download(
                repo_id=name,
                revision="main",
                local_dir=self._dataset_dir(name),
                allow_patterns=list(DATASET_ALLOWED_PATTERNS),
                repo_type="dataset",
            )
        except Exception as e:
            print(f"{Fore.RED}Download error for {name}: {e}{Style.RESET_ALL}")

    def _resolve_config(self, name):
        """
        Pick a config name automatically.
        Uses the saved choice if available, otherwise picks the first one.
        """
        configs = get_dataset_config_names(name)
        print(f"{Fore.CYAN}Available configs for {name}: {configs}{Style.RESET_ALL}")

        if name in self.saved_configs:
            return self.saved_configs[name]

        # Remember the first config for next time
        chosen = configs[0] if configs else None
        self.saved_configs[name] = chosen
        self._save_configs()
        return chosen

    def _pick_split(self, name, config_name):
        """Return 'train' if available, else 'test', else the default."""
        try:
            splits = get_dataset_split_names(name, config_name)
            if "train" in splits:
                return "train"
            if "test" in splits:
                return "test"
        except Exception:
            pass
        return self.split

    def _load_saved_configs(self):
        """Read saved config choices from disk."""
        try:
            if self.saved_config_file.exists():
                with open(self.saved_config_file, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_configs(self):
        """Write current config choices back to disk."""
        with open(self.saved_config_file, "w") as f:
            json.dump(self.saved_configs, f, indent=4)

    def _dataset_dir(self, name):
        """Return (and create) the local directory for this dataset."""
        short_name  = name.split("/")[-1]
        dataset_dir = self.datasets_dir / short_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir


# ── Model loader ───────────────────────────────────────────────────────────────

class ModelLoader:
    """Download and cache a HuggingFace model locally."""

    def __init__(self):
        self.variable       = Variable()
        self.saved_model_dir = self.variable.LocalModel_DIR
        self.saved_model_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self, name):
        """
        Download a model by repo ID or return its local path if cached.

        Returns the local Path object, or None on failure.
        """
        # Already a local path?
        local_path = Path(name)
        if local_path.exists():
            print(f"{Fore.GREEN}Using local model: {local_path}{Style.RESET_ALL}")
            return local_path

        # Already cached in our directory?
        model_dir = self.saved_model_dir / name
        if model_dir.exists():
            print(f"{Fore.GREEN}Model cached: {model_dir}{Style.RESET_ALL}")
            return model_dir

        # Download from HuggingFace
        try:
            model     = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"{Fore.GREEN}Saved model to: {model_dir}{Style.RESET_ALL}")
            return model_dir
        except Exception as e:
            print(f"{Fore.RED}Model load error: {e}{Style.RESET_ALL}")
            self._remove_from_card(name)
            return None

    def _remove_from_card(self, model_name):
        """Remove a failed model entry from ApiCardSet.json."""
        import json
        card_path = self.variable.Card_Path
        try:
            with open(card_path) as f:
                card = json.load(f)
            card["model"].pop(model_name, None)
            with open(card_path, "w") as f:
                json.dump(card, f, indent=4)
        except Exception:
            pass


# ── Orchestrator ───────────────────────────────────────────────────────────────

class DataLoader:
    """
    Download every model and dataset listed in an ApiCard dict.

    Usage (CLI or GUI):
        loader = DataLoader()
        loader.run(api_card_dict)
    """

    def __init__(self):
        self.model_loader   = ModelLoader()
        self.dataset_loader = FlexibleDatasetLoader()

    def run(self, params):
        """Alias for load() — kept for backward compatibility."""
        return self.load(params)

    def load(self, card):
        """
        card – dict with structure {"model": {model_name: {dataset_name: ""}}}
        Downloads each model and all its associated datasets.
        """
        for model_name, datasets in card.get("model", {}).items():
            print(f"Downloading model:    {model_name}")
            print(f"Downloading datasets: {list(datasets.keys())}")
            self._download_model_and_datasets(model_name, datasets)

    def _download_model_and_datasets(self, model_name, datasets):
        """Download a single model and its datasets."""
        try:
            self.model_loader.load_model(model_name)
        except Exception as e:
            print(f"{Fore.RED}Model error [{model_name}]: {e}{Style.RESET_ALL}")
            return

        if not isinstance(datasets, dict):
            return

        for dataset_name in datasets:
            try:
                saved_config = self.dataset_loader.saved_configs.get(dataset_name)
                self.dataset_loader.load(dataset_name, saved_config)
            except Exception as e:
                print(f"{Fore.RED}Dataset error [{dataset_name}]: {e}{Style.RESET_ALL}")


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m modules.DataDownload <repo_id> [--dataset]")
        sys.exit(1)

    repo_id    = sys.argv[1]
    is_dataset = "--dataset" in sys.argv

    if is_dataset:
        loader = FlexibleDatasetLoader()
        loader.load(repo_id)
    else:
        loader = ModelLoader()
        loader.load_model(repo_id)
