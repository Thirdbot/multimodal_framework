"""
ConversationModel.py – Standalone conversation model built on ModelTemplate.

Provides:
  - ConversationConfig : HuggingFace PretrainedConfig for conversation models
  - ConversationModel  : ModelTemplate subclass with embedding + projection layers
                         for text generation (greedy decoding when no inner model)
"""

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..ModelCreationtemplate import ModelTemplate, CustomModelConfig



# ── Config ─────────────────────────────────────────────────────────────────────

class ConversationConfig(PretrainedConfig):
    """HuggingFace config for the conversation-model type."""
    model_type    = "conversation-model"
    architectures = ["ConversationModel"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# ── Model ──────────────────────────────────────────────────────────────────────

class ConversationModel(ModelTemplate):
    """
    Conversation model with simple embedding + projection layers.

    When inner_model is provided the forward/generate calls are delegated to it.
    When inner_model is None, a lightweight embedding → projection head is used
    (intended for from_pretrained loads and quick demos).
    """
    config_class = ConversationConfig

    def __init__(self, config, inner_model=None, tokenizer=None, **kwargs):
        super().__init__(config, inner_model, **kwargs)
        self.tokenizer = tokenizer

        # Lightweight layers used when no inner model is available
        if inner_model is None:
            import torch.nn as nn
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.proj_out  = nn.Linear(config.hidden_size, config.vocab_size)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Embedding → logits forward pass; delegates to inner_model if present."""
        # Move inputs to the model device
        if input_ids      is not None: input_ids      = input_ids.to(self.device)
        if attention_mask is not None: attention_mask = attention_mask.to(self.device)
        if labels         is not None: labels         = labels.to(self.device)

        # Delegate to wrapped model if available
        if self.model is not None:
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
            )

        # Simple forward: embedding → projection → logits
        embeddings = self.embedding(input_ids)   # (batch, seq_len, hidden_size)
        logits     = self.proj_out(embeddings)   # (batch, seq_len, vocab_size)

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

    # ── Generate ───────────────────────────────────────────────────────────────

    def generate(self, input_ids=None, attention_mask=None, max_length=100, **kwargs):
        """Greedy token generation; delegates to inner_model if available."""
        if input_ids is None:
            raise ValueError("input_ids is required")

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Delegate to wrapped model if it supports generation
        if self.model is not None and hasattr(self.model, 'generate'):
            return self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_length=max_length, **kwargs
            )

        # Simple greedy generation loop
        for _ in range(max_length - input_ids.shape[1]):
            outputs    = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            input_ids  = torch.cat([input_ids, next_token], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)], dim=1
                )

        return input_ids

    # ── Save / load ────────────────────────────────────────────────────────────

    def save_pretrained(self, save_directory, **kwargs):
        """Save model weights and tokenizer (if stored)."""
        super().save_pretrained(save_directory, **kwargs)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load model and attempt to load the matching tokenizer."""
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        try:
            model.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        except Exception as e:
            print(f"Warning: could not load tokenizer: {e}")
            model.tokenizer = None
        return model


# ── HuggingFace AutoModel registration ────────────────────────────────────────

AutoConfig.register("conversation-model", ConversationConfig)
AutoModelForCausalLM.register(ConversationConfig, ConversationModel)
