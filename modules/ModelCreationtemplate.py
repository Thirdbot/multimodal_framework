from transformers import PretrainedConfig, PreTrainedModel
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import torch




@dataclass
class ModelConfig:
    """Configuration for custom models."""
    model_name: str = "CustomModel"
    architectures: list[str] = None
    model_type: str = "custom-model"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    vocab_size: int = 50257
    use_cache: bool = False
    gradient_checkpointing: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
   


class CustomModelConfig(PretrainedConfig):
    """Custom model configuration."""
    
    def __init__(self, base_config: Optional[ModelConfig] = None, **kwargs):

        # Extract config values before calling super
        if base_config:
            config_dict = asdict(base_config)
            for key, value in config_dict.items():
                kwargs.setdefault(key, value)
                setattr(self, key, value)
                setattr(CustomModelConfig, key, value)
        
        super().__init__(**kwargs)
      


class ModelTemplate(PreTrainedModel):
    """Template for custom models with overridable forward and generate."""
    config_class = CustomModelConfig
    
    def __init__(self, config: CustomModelConfig, inner_model=None, **kwargs):
        super().__init__(config, **kwargs)
        self.model = inner_model
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Override this method in subclass."""
        if self.model is None:
            raise NotImplementedError("Inner model not set")
        
        if input_ids is not None:
            input_ids = input_ids.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        if labels is not None:
            labels = labels.to(self._device)
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def generate(self, *args, **kwargs):
        """Override this method in subclass."""
        if self.model is None or not hasattr(self.model, 'generate'):
            raise NotImplementedError("Inner model doesn't support generate")
        return self.model.generate(*args, **kwargs)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
    
    def train(self, mode: bool = True):
        super().train(mode)
        if self.model and hasattr(self.model, 'train'):
            self.model.train(mode)
        return self
    
    def eval(self):
        super().eval()
        if self.model and hasattr(self.model, 'eval'):
            self.model.eval()
        return self

def create_model(
    model_name: str,
    model_type: str = "custom",
    inner_model=None,
    architectures: list[str] = None,
    **config_params
) -> ModelTemplate:
    """Create a custom model with parameters."""
    config = ModelConfig(
        model_name=model_name,
        model_type=model_type,
        architectures=architectures,
        **config_params
    )
    model_config = CustomModelConfig(base_config=config)
    return ModelTemplate(model_config, inner_model=inner_model)