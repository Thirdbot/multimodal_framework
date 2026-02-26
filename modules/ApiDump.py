"""
ApiDump.py – ApiCard manager.

ApiCardSet.json records which models pair with which datasets.
The GUI and CLI both call ApiCardSetup.set() to update this file.
"""

import json
import os
from pathlib import Path

from modules.variable import Variable


class ApiCardSetup:
    """Read, update, and save the ApiCardSet.json card."""

    def __init__(self):
        self.variable = Variable()
        self.api_card_path = self.variable.Card_Path

    # ── Public API ─────────────────────────────────────────────────────────────

    def set(self, list_models=None, list_datasets=None, from_repository=None):
        """
        Merge new models / datasets into the ApiCard and save it.

        list_models      – iterable of HuggingFace ModelInfo objects (optional)
        list_datasets    – iterable of HuggingFace DatasetInfo objects (optional)
        from_repository  – dict like {"model": {"model_name": {"dataset": ""}}} (optional)

        Returns the updated card dict.
        """
        model_names   = self._collect_model_names(list_models, from_repository)
        dataset_names = self._collect_dataset_names(list_datasets, from_repository)

        card = self._load_card()
        card = self._merge_into_card(card, model_names, dataset_names)
        self._save_card(card)

        return card

    # ── Private helpers ────────────────────────────────────────────────────────

    def _collect_model_names(self, list_models, from_repository):
        """Return a list of model name strings from all provided sources."""
        names = []
        if list_models is not None:
            names.extend(m.id for m in list_models)
        if from_repository is not None:
            names.extend(from_repository.get("model", {}).keys())
        return names

    def _collect_dataset_names(self, list_datasets, from_repository):
        """Return a list of dataset name strings from all provided sources."""
        names = []
        if list_datasets is not None:
            names.extend(d.id for d in list_datasets)
        if from_repository is not None:
            for model_datasets in from_repository.get("model", {}).values():
                names.extend(model_datasets.keys())
        return names

    def _load_card(self):
        """Load the existing card from disk, or return an empty card."""
        card = {"model": {}}
        path = str(self.api_card_path)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    card = json.load(f)
            except Exception:
                pass  # Corrupted file – start fresh
        return card

    def _merge_into_card(self, card, model_names, dataset_names):
        """
        Add every model → dataset pair to the card.
        Using dicts prevents duplicates automatically.
        """
        for model in model_names:
            if model not in card["model"]:
                card["model"][model] = {}
            for dataset in dataset_names:
                card["model"][model][dataset] = ""
        return card

    def _save_card(self, card):
        """Write the card dict to disk as indented JSON."""
        with open(self.api_card_path, "w") as f:
            json.dump(card, f, ensure_ascii=False, indent=4)


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick test: create / print the card without adding anything new
    setup = ApiCardSetup()
    card  = setup.set()
    print(json.dumps(card, indent=4))
