"""
VisionModel.py – Standalone vision model built on ModelTemplate.

Provides:
  - VisionConfig : HuggingFace PretrainedConfig for vision models
  - VisionModel  : ModelTemplate subclass that combines vision and text embeddings
                   (uses inner_model delegation when one is provided)
"""

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..ModelCreationtemplate import ModelTemplate, CustomModelConfig


# ── Config ─────────────────────────────────────────────────────────────────────

class VisionConfig(PretrainedConfig):
    """HuggingFace config for the vision-model type."""
    model_type    = "vision-model"
    architectures = ["VisionModel"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# ── Model ──────────────────────────────────────────────────────────────────────

class VisionModel(ModelTemplate):
    """
    Vision model that concatenates image and text embeddings.

    When inner_model is provided the forward/generate calls are delegated to it.
    When inner_model is None, a simple vision encoder + text embedding + projection
    head is used (intended for from_pretrained loads and quick demos).
    """
    config_class = VisionConfig

    def __init__(self, config, inner_model=None):
        super().__init__(config, inner_model)

        # Lightweight layers used when no inner model is available
        if inner_model is None:
            import torch.nn as nn
            self.vision_encoder = nn.Linear(1024, config.hidden_size)
            self.text_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.proj_out       = nn.Linear(config.hidden_size, config.vocab_size)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, input_ids=None, attention_mask=None,
                pixel_values=None, labels=None, **kwargs):
        """Combine vision and text embeddings, then project to vocabulary logits."""
        # Move all inputs to the model device
        if input_ids      is not None: input_ids      = input_ids.to(self.device)
        if attention_mask is not None: attention_mask = attention_mask.to(self.device)
        if pixel_values   is not None: pixel_values   = pixel_values.to(self.device)
        if labels         is not None: labels         = labels.to(self.device)

        # Delegate to wrapped model if available
        if self.model is not None:
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, labels=labels, **kwargs
            )

        # Text embeddings
        text_emb = self.text_embedding(input_ids)  # (batch, seq_len, hidden_size)

        if pixel_values is not None:
            # Encode vision tokens and prepend to text
            vision_emb = self.vision_encoder(pixel_values)  # (batch, patches, hidden_size)
            embeddings = torch.cat([vision_emb, text_emb], dim=1)
            if attention_mask is not None:
                vision_mask    = torch.ones(
                    vision_emb.shape[0], vision_emb.shape[1], device=self.device
                )
                attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        else:
            embeddings = text_emb

        logits = self.proj_out(embeddings)  # (batch, total_len, vocab_size)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                reduction='mean',
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # ── Save / load ────────────────────────────────────────────────────────────

    def save_pretrained(self, save_directory, **kwargs):
        """Save model weights and tokenizer (if stored)."""
        super().save_pretrained(save_directory, **kwargs)
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load model and attempt to load the matching tokenizer."""
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        try:
            model.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        except Exception as e:
            print(f"Warning: could not load tokenizer: {e}")
            model.tokenizer = None
        return model


# ── HuggingFace AutoModel registration ────────────────────────────────────────

AutoConfig.register(model_type="vision-model", config=VisionConfig)
AutoModelForCausalLM.register(VisionConfig, VisionModel)
