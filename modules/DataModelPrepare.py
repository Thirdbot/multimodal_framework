"""
DataModelPrepare.py – Tokenize and format datasets for fine-tuning.

Usage (CLI):
    python -m modules.DataModelPrepare

Usage (GUI / code):
    manager = Manager()
    manager.dataset_prepare(api_card_dict)
"""

import json
import os
from pathlib import Path

import torch
from colorama import Fore, Style, init
from datasets import (
    load_dataset,
    concatenate_datasets,
    get_dataset_split_names,
)

from modules.chatTemplate import ChatTemplate
from modules.ModelUtils import load_saved_model
from modules.variable import Variable

init(autoreset=True)

# Suppress OpenMP duplicate lib warnings on some platforms
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Default maximum sequence length for tokenization
DEFAULT_MAX_LENGTH = 1000


class Manager:
    """
    Orchestrates dataset loading, chat-template formatting, and tokenization.

    Call dataset_prepare(api_card) to run the full pipeline and write
    formatted datasets to DATASET_FORMATTED_DIR.
    """

    def __init__(self):
        self.variable = Variable()

        # Resolve device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Ensure output directory exists
        os.makedirs(self.variable.DATASET_FORMATTED_DIR, exist_ok=True)

    # ── Public entry point ─────────────────────────────────────────────────────

    def dataset_prepare(self, list_model_data):
        """
        Format and tokenize every model + dataset pair in api_card.

        list_model_data – dict like {"model": {model_name: {dataset_name: ""}}}

        Saves each formatted dataset to DATASET_FORMATTED_DIR and writes
        training_config.json with column metadata.
        """
        saved_configs = self._load_saved_configs()
        self.variable.training_config_path.touch(exist_ok=True)

        training_config = {"model": {}}

        try:
            for model_name, datasets in list_model_data.get("model", {}).items():
                model_name = Path(model_name)
                model_part = model_name.parts
                split_name =[model_part[-2] , model_part[-1]]

                model_name = model_name.as_posix()
                model_name_sub = Path(*split_name).as_posix()
                # model_path = self.variable.REPO_DIR / "models" / model_name # this one only load from source
                # model_path = self.variable.WORKSPACE / "custom_models" / model_name if len(model_part[0]) != "model" else self.variable.REPO_DIR / "models" / model_name_sub  #Temporal solution for selecting custom model or local model as dataset tokenizer format
                model_path = self.variable.WORKSPACE  / "custom_models" / model_name if model_part[0] != "models" else self.variable.REPO_DIR / model_name  #Temporal solution for selecting custom model or local model as dataset tokenizer format

                model, tokenizer = self._load_model(model_path)

                if model is None or tokenizer is None:
                    print(f"{Fore.RED}Skipping {model_name}: model load failed{Style.RESET_ALL}")
                    continue

                training_config["model"][model_name] = {}
                col_meta = self._process_model_datasets(
                    # model_path originally formatting dataset from source model
                    tokenizer, datasets, saved_configs
                )
                training_config["model"][model_name].update(col_meta)

        except Exception as e:
            print(f"{Fore.RED}dataset_prepare error: {e}{Style.RESET_ALL}")
        finally:
            self._save_training_config(training_config)

    # ── Model helpers ──────────────────────────────────────────────────────────

    def _load_model(self, model_path):
        """Load a model and tokenizer; return (model, tokenizer) or (None, None)."""
        print(f"{Fore.CYAN}Loading model: {model_path}{Style.RESET_ALL}")
        try:
            return load_saved_model(model_path)
        except Exception as e:
            print(f"{Fore.RED}Model load error [{model_path}]: {e}{Style.RESET_ALL}")
            return None, None

    # ── Dataset processing ─────────────────────────────────────────────────────

    def _process_model_datasets(self, tokenizer, datasets, saved_configs):
        """
        Process all datasets for one model.

        Returns a dict of {dataset_name: column_meta} for training_config.json.
        """
        col_meta    = {}
        first_ds    = None
        first_cols  = set()

        for dataset_name in datasets:
            print(f"{Fore.CYAN}Processing dataset: {dataset_name}{Style.RESET_ALL}")

            raw_dataset = self._load_dataset(dataset_name, saved_configs.get(dataset_name))
            if raw_dataset is None:
                continue

            formatted = self._apply_template(dataset_name, tokenizer, raw_dataset, tokenizing=False)
            if formatted is None:
                print(f"{Fore.RED}Template failed for: {dataset_name}{Style.RESET_ALL}")
                continue

            if first_ds is None:
                first_ds   = formatted
                first_cols = set(formatted.column_names)
                concat_ds  = formatted
            else:
                second_cols = set(formatted.column_names)
                concat_ds, first_ds = self._merge_datasets(first_ds, first_cols, formatted, second_cols)
                first_cols = set(concat_ds.column_names)

            # Tokenize the current (possibly concatenated) dataset
            tokenized = self._apply_template(dataset_name, tokenizer, concat_ds, tokenizing=True)
            if tokenized is None:
                continue

            formatted_name = self._save_formatted_dataset(tokenized, dataset_name, prefix="")
            col_meta[dataset_name] = formatted_name

        return col_meta

    def _load_dataset(self, dataset_name, config_name=None):
        """Load a dataset from the local repository directory."""
        local_path = self.variable.DATASETS_DIR / dataset_name
        path_str   = local_path.as_posix()
        print(f"{Fore.CYAN}Loading dataset: {path_str}  config={config_name}{Style.RESET_ALL}")

        try:
            splits = get_dataset_split_names(path_str, config_name)
            split  = "train" if "train" in splits else ("test" if "test" in splits else "train")
            return load_dataset(path_str, split=split)
        except Exception as e:
            print(f"{Fore.RED}Dataset load error [{dataset_name}]: {e}{Style.RESET_ALL}")
            return None

    def _apply_template(self, dataset_name, tokenizer, dataset, tokenizing=False):
        """Apply the chat template / tokenizer to a dataset."""
        # Ensure pad token is set before template processing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            template = ChatTemplate(tokenizer=tokenizer)
            result = template.prepare_dataset(
                dataset_name, dataset,
                max_length=DEFAULT_MAX_LENGTH,
                Tokenizing=tokenizing,
            )
            return result
        except Exception as e:
            print(f"{Fore.RED}Template error [{dataset_name}]: {e}{Style.RESET_ALL}")
            return None

    def _merge_datasets(self, first_ds, first_cols, second_ds, second_cols):
        """
        Align column sets and concatenate two datasets.

        Missing columns are filled with None so both datasets share the same schema.
        """
        # Add missing columns to each side
        for col in second_cols - first_cols:
            first_ds = first_ds.add_column(col, [None] * len(first_ds))
        for col in first_cols - second_cols:
            second_ds = second_ds.add_column(col, [None] * len(second_ds))

        concat = concatenate_datasets([first_ds, second_ds])
        print(f"{Fore.GREEN}Merged columns: {concat.column_names}{Style.RESET_ALL}")
        return concat, first_ds

    def _save_formatted_dataset(self, dataset, dataset_name, prefix=""):
        """Save a formatted dataset to disk and return the folder name."""
        safe_name = dataset_name.replace("/", "_") + "_formatted"
        if prefix:
            safe_name = f"{prefix}_{safe_name}"

        out_path = self.variable.DATASET_FORMATTED_DIR / safe_name
        out_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(out_path))
        print(f"{Fore.GREEN}Saved: {out_path}{Style.RESET_ALL}")
        return safe_name

    # ── Config helpers ─────────────────────────────────────────────────────────

    def _load_saved_configs(self):
        """Load saved dataset config choices (name → config_name)."""
        path = self.variable.SAVED_CONFIG_Path.as_posix()
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_training_config(self, config):
        """Write training_config.json with the column metadata."""
        with open(self.variable.training_config_path, "w") as f:
            json.dump(config, f, indent=4)


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json

    # Read the api card from ApiCardSet.json and run dataset_prepare
    v = Variable()
    if not v.Card_Path.exists():
        print("ApiCardSet.json not found. Run ApiDump.py first.")
        sys.exit(1)

    with open(v.Card_Path) as f:
        card = json.load(f)

    manager = Manager()
    manager.dataset_prepare(card)
