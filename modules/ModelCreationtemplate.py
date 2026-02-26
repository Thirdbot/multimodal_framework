"""
ModelCreationtemplate.py – Base classes for custom model architectures.

Provides:
  - ModelConfig      : simple dataclass for architecture parameters
  - CustomModelConfig: HuggingFace-compatible PretrainedConfig wrapper
  - ModelTemplate    : PreTrainedModel base with overridable forward / generate
  - create_model()   : factory helper
"""

from dataclasses import dataclass, asdict
from transformers import PretrainedConfig, PreTrainedModel
import torch


# ── Configuration dataclass ────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Architecture parameters for a custom model."""
    model_name:           str       = "CustomModel"
    model_type:           str       = "custom-model"
    hidden_size:          int       = 768
    num_hidden_layers:    int       = 12
    num_attention_heads:  int       = 12
    vocab_size:           int       = 50257
    use_cache:            bool      = False
    gradient_checkpointing: bool    = True
    architectures:        list      = None  # e.g. ["ConversationModel"]

    def to_dict(self):
        """Return a plain dict representation."""
        return asdict(self)


# ── HuggingFace-compatible config ──────────────────────────────────────────────

class CustomModelConfig(PretrainedConfig):
    """
    Wraps ModelConfig as a HuggingFace PretrainedConfig so the model
    can be saved and loaded with save_pretrained / from_pretrained.
    """

    def __init__(self, base_config=None, **kwargs):
        # Fold dataclass fields into kwargs before calling super()
        if base_config is not None:
            for key, value in asdict(base_config).items():
                kwargs.setdefault(key, value)
                setattr(self, key, value)
        super().__init__(**kwargs)


# ── Base model template ────────────────────────────────────────────────────────

class ModelTemplate(PreTrainedModel):
    """
    Abstract base for all custom models.

    Subclasses override forward() and optionally generate().
    An inner_model can be passed to delegate computation to an existing
    HuggingFace model.
    """
    config_class = CustomModelConfig

    def __init__(self, config, inner_model=None, **kwargs):
        super().__init__(config, **kwargs)
        self.model   = inner_model  # Optional wrapped/base model
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Forward / generate ─────────────────────────────────────────────────────

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Default forward: move tensors to device and delegate to inner_model.
        Override in subclasses for custom architectures.
        """
        if self.model is None:
            raise NotImplementedError("No inner model. Override forward() in your subclass.")

        if input_ids      is not None: input_ids      = input_ids.to(self._device)
        if attention_mask is not None: attention_mask = attention_mask.to(self._device)
        if labels         is not None: labels         = labels.to(self._device)

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(self, *args, **kwargs):
        """Delegate generation to inner_model."""
        if self.model is None or not hasattr(self.model, "generate"):
            raise NotImplementedError("inner_model does not support generate()")
        return self.model.generate(*args, **kwargs)

    # ── Gradient checkpointing ─────────────────────────────────────────────────

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    # ── Train / eval mode ─────────────────────────────────────────────────────

    def train(self, mode=True):
        super().train(mode)
        if self.model:
            self.model.train(mode)
        return self

    def eval(self):
        super().eval()
        if self.model:
            self.model.eval()
        return self


# ── Factory function ───────────────────────────────────────────────────────────

def create_model(model_name, model_type="custom", inner_model=None,
                 architectures=None, **config_params):
    """
    Convenience factory to create a ModelTemplate instance.

    model_name    – human-readable model name
    model_type    – type string registered with HuggingFace AutoModel
    inner_model   – optional existing model to wrap
    architectures – list of class name strings for HuggingFace registry
    config_params – extra parameters merged into ModelConfig
    """
    config = ModelConfig(
        model_name=model_name,
        model_type=model_type,
        architectures=architectures,
        **config_params,
    )
    hf_config = CustomModelConfig(base_config=config)
    return ModelTemplate(hf_config, inner_model=inner_model)
