from ..ModelCreationtemplate import ModelTemplate, CustomModelConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig,AutoTokenizer
import torch.nn.functional as F
import torch


# Config class for conversation model
class ConversationConfig(PretrainedConfig):
    model_type = "conversation-model"
    architectures = ["ConversationModel"]

    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ConversationModel(ModelTemplate):
    """Example conversation model with simple forward logic."""
    config_class = ConversationConfig
    
    def __init__(self, config: CustomModelConfig,inner_model=None, tokenizer=None, **kwargs):
        # Extract inner_model from kwargs - this makes it truly optional
        # inner_model = kwargs.pop('inner_model', None)
        # Handle both from_pretrained (no inner_model) and direct instantiation
        super().__init__(config, inner_model, **kwargs)
        # Store tokenizer
        self.tokenizer = tokenizer
        # Simple embedding and projection layers for demo
        if inner_model is None:
            import torch.nn as nn
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.proj_out = nn.Linear(config.hidden_size, config.vocab_size)

    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Simple forward pass with embedding + projection."""
        
        # Move inputs to device
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        # Use inner model if available
        if self.model is not None:
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        
        # Simple forward: embedding -> projection -> logits
        embeddings = self.embedding(input_ids)  # (batch, seq_len, hidden_size)
        logits = self.proj_out(embeddings)      # (batch, seq_len, vocab_size)
        
        loss = None
        if labels is not None:
            # Calculate language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), reduction='mean')
        
        return CausalLMOutputWithPast(loss=loss, logits=logits)
    
    def generate(self, input_ids=None, attention_mask=None, max_length=100, **kwargs):
        """Simple greedy text generation."""
        if input_ids is None:
            raise ValueError("input_ids required")
        
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Use inner model if available
        if self.model is not None and hasattr(self.model, 'generate'):
            return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, **kwargs)
        
        # Simple greedy generation
        for _ in range(max_length - input_ids.shape[1]):
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # Get last token logits
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # Greedy
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        
        return input_ids
    
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

        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        # Try to load tokenizer from the same directory
        try:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
            model.tokenizer = tokenizer
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            model.tokenizer = None
        
        return model




# # Register ConversationModel with AutoModelForCausalLM for the conversation-model type
AutoConfig.register("conversation-model", ConversationConfig)
AutoModelForCausalLM.register(ConversationConfig, ConversationModel)