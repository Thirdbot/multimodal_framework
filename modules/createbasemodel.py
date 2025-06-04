import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#for create base model to be use for finetuning and customize architecture
from transformers import LlamaModel, LlamaConfig, AutoTokenizer, AutoConfig
import torch
from pathlib import Path


#goal is to make a tokenizer and a model and save it to the models directory

#simple model backbone using llama (customize later)
class CreateModel:
    def __init__(self, model_name, model_category):
        self.model_name = model_name
        self.model_id = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
        self.model_category = model_category
        
        self.model_path = Path(__file__).parent.parent.absolute() / "custom_models" / self.model_category / self.model_name
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # First load the tokenizer to get the correct vocab size
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True
        )
        
        # Get the original config to maintain compatibility
        original_config = AutoConfig.from_pretrained(self.model_id)
        
        # Create modified config based on original
        self.BaseLlamaConfig = LlamaConfig(
            vocab_size=len(self.tokenizer),  # Use actual vocab size
            hidden_size=2048,
            num_hidden_layers=16,
            num_attention_heads=16,
            intermediate_size=5504,
            max_position_embeddings=original_config.max_position_embeddings,  # Keep original context length
            torch_dtype=torch.float16,
            use_cache=False,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Initialize model with memory optimizations
        self.baseLlamaModel = LlamaModel(
            self.BaseLlamaConfig
            )
        
        # Customize and save in one go
        self.save_model()
        
    def customeize_model(self):
        """Customize the model architecture."""
        base_model = self.baseLlamaModel
        # Add your customizations here
        return base_model
    def save_model(self):
        """Save the model and tokenizer with optimizations."""
        model = self.customeize_model()
        
        # Save model configuration
        self.BaseLlamaConfig.save_pretrained(self.model_path)
        
        # Save with optimizations
        model.save_pretrained(
            self.model_path,
            max_shard_size="500MB",
            safe_serialization=True
        )
        
        self.tokenizer.save_pretrained(
            self.model_path,
            legacy_format=False
        )
