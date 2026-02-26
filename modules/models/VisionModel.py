from ..ModelCreationtemplate import ModelTemplate,CustomModelConfig,PretrainedConfig
from transformers import AutoConfig,AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn.functional as F

class VisionConfig(PretrainedConfig):
    model_type = "vision-model"
    architectures = ["VisionModel"]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
class VisionModel(ModelTemplate):
    """Example vision model with simple forward logic."""
    # Register the config at class level
    config_class = VisionConfig
    def __init__(self, config: CustomModelConfig, inner_model=None):
        super().__init__(config, inner_model)
        # Simple vision layers for demo
        if inner_model is None:
            import torch.nn as nn
            self.vision_encoder = nn.Linear(1024, config.hidden_size)  # Assume 1024D vision input
            self.text_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.proj_out = nn.Linear(config.hidden_size, config.vocab_size)
        else:
            # if model already have vision encoder, text embedding, proj_out,
            # if model not have, create simple ones  basically from old add_vision
            pass
    
    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None, **kwargs):
        """Simple forward combining vision and text."""
        
        # Move all inputs to device
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        # Use inner model if available
        if self.model is not None:
            return self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels, **kwargs)
        
        # Simple vision + text forward
        text_emb = self.text_embedding(input_ids)  # (batch, seq_len, hidden_size)
        
        if pixel_values is not None:
            # Simple vision encoding
            vision_emb = self.vision_encoder(pixel_values)  # (batch, num_patches, hidden_size)
            # Concatenate vision and text
            embeddings = torch.cat([vision_emb, text_emb], dim=1)
            if attention_mask is not None:
                # Extend attention mask for vision tokens
                vision_mask = torch.ones(vision_emb.shape[0], vision_emb.shape[1], device=self.device)
                attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        else:
            embeddings = text_emb
        
        # Project to vocabulary
        logits = self.proj_out(embeddings)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), reduction='mean')
        
        return CausalLMOutputWithPast(loss=loss, logits=logits)
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save model and tokenizer."""
        # Save the model itself
        super().save_pretrained(save_directory, **kwargs)
        
        # Save tokenizer if stored
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load model and tokenizer."""
        from transformers import AutoTokenizer
        
        # Load the model
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # Try to load tokenizer from the same directory
        try:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
            model.tokenizer = tokenizer
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            model.tokenizer = None
        
        return model



        
# Register ConversationModel with AutoModelForCausalLM for the vision-model type
AutoConfig.register(model_type="vision-model",config=VisionConfig)
AutoModelForCausalLM.register(VisionConfig, VisionModel)