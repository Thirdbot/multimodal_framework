"""
train.py – Fine-tune a saved custom model on formatted datasets.

Usage (CLI):
    python -m modules.train

Usage (GUI / code):
    ft = FinetuneModel()
    ft.finetune_model()
"""

import json
import os
from pathlib import Path
import torch
from colorama import Fore, Style
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments, AutoModelForCausalLM,
)

from modules.ModelUtils import load_saved_model
from modules.variable import Variable

# ── Environment tweaks ─────────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"]    = "0"


# ── Vision-aware data collator ─────────────────────────────────────────────────

class VisionDataCollator:
    """
    Custom collator that handles multimodal (image + text) batches.

    The standard DataCollatorForLanguageModeling strips pixel_values,
    so we use this wrapper whenever the dataset contains images.
    """

    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer           = tokenizer
        self.pad_to_multiple_of  = pad_to_multiple_of
        self.padding_side        = tokenizer.padding_side

    def __call__(self, features):
        """Collate a list of feature dicts into a padded batch."""
        has_images  = "pixel_values" in features[0]
        input_ids   = [f["input_ids"]      for f in features]
        attn_masks  = [f["attention_mask"] for f in features]

        # Use explicit labels if present, otherwise copy input_ids
        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
        else:
            labels = [
                f["input_ids"].clone() if isinstance(f["input_ids"], torch.Tensor)
                else list(f["input_ids"])
                for f in features
            ]

        max_len = max(
            len(ids) if isinstance(ids, list) else ids.shape[0]
            for ids in input_ids
        )

        batch = self._pad_sequences(input_ids, attn_masks, labels, max_len)

        # Stack pixel tensors if present
        if has_images:
            pv = [f["pixel_values"] for f in features]
            batch["pixel_values"] = (
                torch.stack(pv) if isinstance(pv[0], torch.Tensor)
                else torch.stack([torch.tensor(p) for p in pv])
            )

        return batch

    def _pad_sequences(self, input_ids, attn_masks, labels, max_len):
        """Pad all sequences to max_len and return a dict of tensors."""
        n   = len(input_ids)
        pad = self.tokenizer.pad_token_id

        padded_ids   = torch.full((n, max_len), pad, dtype=torch.long)
        padded_mask  = torch.zeros((n, max_len), dtype=torch.long)
        padded_labels= torch.full((n, max_len), -100, dtype=torch.long)

        for i, (ids, mask, lab) in enumerate(zip(input_ids, attn_masks, labels)):
            ids  = ids  if isinstance(ids,  torch.Tensor) else torch.tensor(ids)
            mask = mask if isinstance(mask, torch.Tensor) else torch.tensor(mask)
            lab  = lab  if isinstance(lab,  torch.Tensor) else torch.tensor(lab)
            slen = len(ids)

            if self.padding_side == "right":
                padded_ids[i,    :slen] = ids
                padded_mask[i,   :slen] = mask
                padded_labels[i, :slen] = lab
            else:
                padded_ids[i,    -slen:] = ids
                padded_mask[i,   -slen:] = mask
                padded_labels[i, -slen:] = lab

        return {
            "input_ids":      padded_ids,
            "attention_mask": padded_mask,
            "labels":         padded_labels,
        }


# ── Main fine-tuner ────────────────────────────────────────────────────────────

class FinetuneModel:
    """
    Orchestrates the full fine-tuning pipeline:
      1. Read training_config.json to know which model/dataset pairs to train
      2. Load each formatted dataset from disk
      3. Load the model via load_saved_model()
      4. Set up Trainer with appropriate data collator
      5. Train and save the checkpoint
    """

    def __init__(self):
        self.variable = Variable()

        # ── Training hyper-parameters ──────────────────────────────────
        self.batch_size                 = 1
        self.gradient_accumulation_steps = 1
        self.learning_rate              = 1e-3
        self.num_train_epochs           = 0.1

        # ── Device ────────────────────────────────────────────────────
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.cuda_available = torch.cuda.is_available()

        # Ensure checkpoint directory exists
        self.variable.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def finetune_model(self):
        """
        Run fine-tuning for every (model, dataset) pair listed in
        training_config.json.
        """
        config = self._load_training_config()
        if config is None:
            return

        for model_name, datasets in config.get("model", {}).items():
            print(f"{Fore.CYAN}Fine-tuning: {model_name}{Style.RESET_ALL}")
            self._train_all_datasets(model_name, datasets)

    # ── Config loading ─────────────────────────────────────────────────────────

    def _load_training_config(self):
        """Read training_config.json; return dict or None on failure."""
        try:
            with open(self.variable.training_config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"{Fore.RED}Could not load training config: {e}{Style.RESET_ALL}")
            return None

    # ── Per-model training ─────────────────────────────────────────────────────

    def _train_all_datasets(self, model_name, datasets):
        """Load and fine-tune the model on every listed dataset."""
        for dataset_name, dataset_info in datasets.items():
            dataset = self._load_formatted_dataset(dataset_name)
            if dataset is None:
                continue

            model, tokenizer, task = self._load_model_for_training(
                model_name, dataset_info
            )
            if model is None:
                continue

            self._clear_gpu_cache()
            self._run_training(model, tokenizer, dataset, model_name, task)

    def _load_formatted_dataset(self, dataset_name):
        """Load a pre-formatted dataset from DATASET_FORMATTED_DIR."""
        safe_name = dataset_name.replace("/", "_") + "_formatted"
        path = self.variable.DATASET_FORMATTED_DIR / safe_name
        try:
            ds = load_from_disk(str(path))
            print(f"{Fore.CYAN}Dataset: {len(ds)} rows{Style.RESET_ALL}")
            return ds
        except Exception as e:
            print(f"{Fore.RED}Dataset load error [{dataset_name}]: {e}{Style.RESET_ALL}")
            return None

    def _load_model_for_training(self, model_name, dataset_info):
        """
        Select the correct model path (local / custom / checkpoint) and task type.

        When dataset_info is a dict with explicit type keys ("conversations", "image",
        "images") those keys are used directly.  When dataset_info is a plain string
        (format written by training_config.json) the method falls back to trying
        conversation load first, then vision, and infers the task from the loaded
        model's config.

        Returns (model, tokenizer, task_name) or (None, None, None).
        """
        model = tokenizer = task = None

        # temporal fix almost same implementation from train_config.json saved to extract the parent's name of model
        model_name = Path(model_name)
        model_part = model_name.parts
        split_name =[model_part[-2] , model_part[-1]]
        model_name = model_name.as_posix()

        model_name_sub = Path(*split_name).as_posix()

        print(model_name_sub)



        # ── Explicit type hints (dataset_info is a dict with known keys) ──────
        # if isinstance(dataset_info, dict):
        #     if "conversations" in dataset_info:
        #         task = "text-generation"
        #
        #         model, tokenizer = self._load_conversation_model(model_name)
        #     if "image" in dataset_info or "images" in dataset_info:
        #         task = "text-vision-text-generation"
        #         model, tokenizer = self._load_vision_model(model_name)
        # else:

        type_part = [model_part[0]]

        model_type = Path(*type_part).as_posix()

        if model_type == "conversation-model":
            model, tokenizer = self._load_conversation_model(model_name_sub)
        elif model_type == "vision-model":
            model, tokenizer = self._load_vision_model(model_name_sub)
        else:
            # localModel if i have mind to implement this part
            local_model =(self.variable.LocalModel_DIR / model_name_sub).as_posix()
            model , tokenizer = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_model)

        # # ── Fallback: try conversation then vision, detect task from config ───
        # if model is None:
        #     model, tokenizer = self._load_conversation_model(model_name_sub)
        # if model is None:
        #     model, tokenizer = self._load_vision_model(model_name_sub)

        # Infer task from the loaded model's config when not already set
        if model is not None and task is None:
            model_type = getattr(getattr(model, "config", None), "model_type", "")
            if "vision" in model_type.lower():
                task = "text-vision-text-generation"
            else:
                task = "text-generation"

        if model is None:
            print(f"{Fore.RED}No model loaded for: {model_name}{Style.RESET_ALL}")
        else:
            model.train()

        return model, tokenizer, task

    def _load_conversation_model(self, model_name):
        """Try local → custom → checkpoint path for conversation models."""
        safe_name  = model_name.replace("/", "_")
        local_path = self.variable.LocalModel_DIR  / model_name
        custom_path= self.variable.REGULAR_MODEL_DIR / model_name
        ckpt_path  = self.variable.CHECKPOINT_DIR / f"text-generation{safe_name}"

        # Load from the most refined checkpoint first
        for path in [ckpt_path, custom_path, local_path]:
            if path.exists():
                label = str(path).split("/")[-1]
                print(f"{Fore.GREEN}Loading conversation model from: {label}{Style.RESET_ALL}")
                return load_saved_model(path)

        print(f"{Fore.RED}No conversation model found for: {model_name}{Style.RESET_ALL}")
        return None, None

    def _load_vision_model(self, model_name):
        """Try checkpoint → custom path for vision models."""
        safe_name  = model_name.replace("/", "_")
        ckpt_path  = self.variable.CHECKPOINT_DIR / f"text-vision-text-generation{safe_name}"
        custom_path= self.variable.VISION_MODEL_DIR / model_name

        if ckpt_path.exists():
            print(f"{Fore.GREEN}Loading vision model from checkpoint{Style.RESET_ALL}")
            model, tokenizer = load_saved_model(ckpt_path)
        else:
            model, tokenizer = load_saved_model(custom_path)

        # Keep the vision adapter trainable
        if model and hasattr(model, "vision_adapter"):
            for param in model.vision_adapter.parameters():
                param.requires_grad = True

        return model, tokenizer

    # ── Trainer setup ──────────────────────────────────────────────────────────

    def _make_training_args(self, task, model_name):
        """Build TrainingArguments for the given task and model name."""
        # Normalise model name to a safe folder name
        # safe_name  = model_name.split("\\")[-1] if "custom_models" in model_name else model_name
        safe_name  = model_name.replace("/", "_") if "/" in model_name else model_name
        output_dir = self.variable.CHECKPOINT_DIR / task / safe_name
        output_dir.mkdir(parents=True, exist_ok=True)

        return TrainingArguments(
            output_dir                     = str(output_dir),
            learning_rate                  = self.learning_rate,
            per_device_train_batch_size    = self.batch_size,
            num_train_epochs               = self.num_train_epochs,
            weight_decay                   = 0.01,
            save_strategy                  = "steps",
            save_steps                     = 10,
            save_total_limit               = 1,
            logging_dir                    = str(output_dir),
            logging_strategy               = "steps",
            logging_steps                  = 5,
            logging_first_step             = True,
            gradient_accumulation_steps    = self.gradient_accumulation_steps,
            fp16                           = False,
            bf16                           = self.cuda_available,
            optim                          = "adamw_8bit" if self.cuda_available else "adamw_torch",
            lr_scheduler_type              = "cosine",
            warmup_ratio                   = 0.01,
            remove_unused_columns          = False,
            label_names                    = ["labels"],
            gradient_checkpointing         = True,
            gradient_checkpointing_kwargs  = {"use_reentrant": False},
            ddp_find_unused_parameters     = False,
            ddp_bucket_cap_mb              = 50,
            dataloader_pin_memory          = False,
            dataloader_num_workers         = 0,
            max_grad_norm                  = 0.5,
            group_by_length                = False,
            report_to                      = "none",
            resume_from_checkpoint         = True,
            save_safetensors               = True,
            save_only_model                = True,
            overwrite_output_dir           = True,
            torch_compile                  = False,
            use_mps_device                 = False,
            eval_strategy                  = "no",
            do_eval                        = False,
            auto_find_batch_size           = False,
            dataloader_prefetch_factor     = None,
        )

    def _pick_data_collator(self, model, dataset, tokenizer):
        """Return the appropriate data collator for the model type."""
        # Vision model: check for pixel_values in training split
        train_split = dataset["train"] if hasattr(dataset, "__getitem__") else dataset
        has_images  = (
            "pixel_values" in train_split.features
            if hasattr(train_split, "features")
            else False
        )
        is_vision   = has_images or (
            hasattr(model, "config") and
            "vision" in getattr(model.config, "model_type", "").lower()
        )

        if is_vision:
            print(f"{Fore.GREEN}Using VisionDataCollator{Style.RESET_ALL}")
            return VisionDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

        print(f"{Fore.CYAN}Using standard DataCollatorForLanguageModeling{Style.RESET_ALL}")
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
        )

    # ── Training execution ─────────────────────────────────────────────────────

    def _run_training(self, model, tokenizer, dataset, model_name, task):
        """Set up and execute one Trainer.train() run."""
        try:
            model.train()
            self._log_param_counts(model)

            train_ds     = dataset["train"]
            collator     = self._pick_data_collator(model, dataset, tokenizer)
            training_args= self._make_training_args(task, model_name)

            print(f"{Fore.CYAN}Training dataset rows: {len(train_ds)}{Style.RESET_ALL}")

            trainer = Trainer(
                model         = model,
                args          = training_args,
                train_dataset = train_ds,
                data_collator = collator,
            )
            trainer.train()

            self._save_model(model, tokenizer, trainer, model_name, task)

        except Exception as e:
            print(f"{Fore.RED}Training error: {e}{Style.RESET_ALL}")
        finally:
            # Free memory even on error
            try:
                del trainer, model
            except Exception:
                pass
            self._clear_gpu_cache()

    def _save_model(self, model, tokenizer, trainer, model_name, task):
        """Save the model to the correct checkpoint subdirectory."""
        model_type  = getattr(getattr(model, "config", None), "model_type", "conversation-model")
        safe_name   = model_name.replace("/", "_") if "/" in model_name else model_name

        if "vision" in model_type.lower():
            save_path = self.variable.CHECKPOINT_DIR / "text-vision-text-generation" / safe_name
            save_path.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(save_path))
            tokenizer.save_pretrained(str(save_path))
            self._save_vision_submodels(model, save_path, tokenizer)
        else:
            save_path = self.variable.CHECKPOINT_DIR / "text-generation" / safe_name
            save_path.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(save_path))
            tokenizer.save_pretrained(str(save_path))
            model.config.save_pretrained(str(save_path))

        print(f"{Fore.GREEN}Model saved: {save_path}{Style.RESET_ALL}")

    def _save_vision_submodels(self, model, save_path, tokenizer):
        """Save the language sub-model, vision sub-model, and adapter separately."""
        import torch

        if hasattr(model, "lang_model"):
            lang_path = save_path / "lang_model"
            lang_path.mkdir(parents=True, exist_ok=True)
            model.lang_model.save_pretrained(str(lang_path))
            tokenizer.save_pretrained(str(lang_path))

        if hasattr(model, "vision_model"):
            vis_path = save_path / "vision_model"
            vis_path.mkdir(parents=True, exist_ok=True)
            model.vision_model.save_pretrained(str(vis_path))

        if hasattr(model, "vision_adapter"):
            adap_path = save_path / "vision_adapter"
            adap_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.vision_adapter.state_dict(),
                       str(adap_path / "vision_adapter.pt"))

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _log_param_counts(self, model):
        """Print trainable vs total parameter counts."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        pct       = 100 * trainable / total if total else 0
        print(f"{Fore.CYAN}Trainable: {trainable:,}  Total: {total:,}  ({pct:.2f}%){Style.RESET_ALL}")

    def _clear_gpu_cache(self):
        """Empty CUDA cache if a GPU is available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ft = FinetuneModel()
    ft.finetune_model()
