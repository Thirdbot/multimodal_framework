import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from login.huggingface_login import HuggingFaceLogin
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from pathlib import Path
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map, disk_offload
from accelerate.utils import set_seed
from models.finetuning_model import FinetuneModel
from typing import Dict, Optional, List
import json
from datetime import datetime
import yaml

# Define paths
WORKSPACE_DIR = Path(__file__).parent.absolute()
MODEL_DIR = WORKSPACE_DIR / "models" / "Text-Text-generation"
OFFLOAD_DIR = WORKSPACE_DIR / "offload"

class PromptConfig:
    """Configuration class for managing prompts using a table-based approach"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Default prompt configurations
        self.prompt_table = {
            "system": {
                "default": "You are a helpful AI assistant. You provide clear, concise, and accurate responses.",
                "creative": "You are a creative AI assistant. You provide imaginative and engaging responses.",
                "technical": "You are a technical AI assistant. You provide detailed and accurate technical information.",
                "friendly": "You are a friendly AI assistant. You provide warm and approachable responses."
            },
            "user": {
                "default": "{input}",
                "question": "Question: {input}",
                "instruction": "Instruction: {input}",
                "chat": "User: {input}"
            },
            "assistant": {
                "default": "Assistant: {response}",
                "answer": "Answer: {response}",
                "response": "Response: {response}",
                "chat": "Assistant: {response}"
            },
            "context": {
                "default": "{history}\n\n{current_prompt}",
                "minimal": "{current_prompt}",
                "full": "Previous conversation:\n{history}\n\nCurrent interaction:\n{current_prompt}"
            }
        }
        
        # Load custom configuration if provided
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
        
        # Set default configurations
        self.current_config = {
            "system": "default",
            "user": "default",
            "assistant": "default",
            "context": "default"
        }
    
    def load_config(self, config_path: str) -> None:
        """Load prompt configurations from a YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                custom_config = yaml.safe_load(f)
                self.prompt_table.update(custom_config)
        except Exception as e:
            print(f"Warning: Could not load custom prompt configuration: {e}")
    
    def set_config(self, config_type: str, config_name: str) -> None:
        """Set a specific configuration type"""
        if config_type in self.current_config and config_name in self.prompt_table[config_type]:
            self.current_config[config_type] = config_name
        else:
            raise ValueError(f"Invalid configuration: {config_type}={config_name}")
    
    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        """Get a formatted prompt based on current configuration"""
        if prompt_type not in self.current_config:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
        
        template = self.prompt_table[prompt_type][self.current_config[prompt_type]]
        return template.format(**kwargs)
    
    def get_available_configs(self) -> Dict[str, List[str]]:
        """Get all available configurations"""
        return {k: list(v.keys()) for k, v in self.prompt_table.items()}
    
    def save_config(self, config_path: str) -> None:
        """Save current prompt configurations to a YAML file"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.prompt_table, f, allow_unicode=True)
        except Exception as e:
            print(f"Warning: Could not save prompt configuration: {e}")

class Conversation:
    """Class to manage conversation history using Hugging Face chat template style"""
    def __init__(self):
        self.messages = []
        self.system_message = {
            "role": "system",
            "content": "You are a helpful AI assistant. You provide clear, concise, and accurate responses."
        }
        self.add_message(self.system_message)
    
    def add_message(self, message: Dict) -> None:
        """Add a message to the conversation history"""
        message["timestamp"] = datetime.now().isoformat()
        self.messages.append(message)
    
    def get_chat_template(self) -> List[Dict]:
        """Get messages in Hugging Face chat template format"""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]
    
    def save_conversation(self, filepath: str) -> None:
        """Save conversation history to a file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save conversation: {e}")
    
    def load_conversation(self, filepath: str) -> None:
        """Load conversation history from a file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_messages = json.load(f)
                # Keep the system message and add loaded messages
                self.messages = [self.messages[0]]  # Keep system message
                self.messages.extend(loaded_messages[1:])  # Add loaded messages
        except FileNotFoundError:
            print(f"No previous conversation found at {filepath}")
        except Exception as e:
            print(f"Warning: Could not load conversation: {e}")
            # Reset to initial state if loading fails
            self.messages = [self.messages[0]]  # Keep only system message

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    def load_model(self, model_path):
        """Load a finetuned model directly without offloading"""
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        
        # Configure quantization for full GPU usage
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            use_fast=True
        )
        
        # Load model directly to GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=True,
            low_cpu_mem_usage=True
        )
        
        # Move model to GPU if not already there
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        
        return self.model
    
    def generate_response_stream(self, user_input: str, max_new_tokens: int = 150, 
                               temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate a response token by token with streaming"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded first")
        
        # Tokenize input with attention mask
        inputs = self.tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Initialize generation parameters
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3
        }
        
        # Generate response token by token
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Generate next token
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    **generation_config
                )
                
                # Get the new token
                new_token_id = outputs[0][-1].item()
                
                # Check if we've reached the end of the sequence
                if new_token_id == self.tokenizer.eos_token_id:
                    break
                
                # Decode and yield the new token
                new_token = self.tokenizer.decode([new_token_id], skip_special_tokens=True)
                yield new_token
                
                # Update input_ids and attention_mask for next iteration
                input_ids = outputs
                attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=self.device)], dim=1)

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Check if model exists
    if not (MODEL_DIR.exists() and any(MODEL_DIR.iterdir())):
        print("No finetuned model found. Starting finetuning process...")
        
        # Initialize finetuner
        finetuner = FinetuneModel(
            model_id="prometheus-eval/prometheus-7b-v2.0",
            dataset_name="oscar-corpus/OSCAR-2201",
            language='th',
            split='train'
        )
        
        # Finetune model
        model_path = finetuner.finetune()
        print(f"Model finetuned and saved to: {model_path}")
    else:
        print("Found existing finetuned model.")
        model_path = MODEL_DIR
    
    # Load the model
    print("Loading model...")
    model_manager.load_model(model_path)
    print("Model loaded successfully!")
    print(f"Model device: {model_manager.device}")
    
    print("\nChat Interface")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    # Interactive loop
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        print("\nAssistant: ", end='', flush=True)
        
        # Generate and stream response token by token
        try:
            for token in model_manager.generate_response_stream(user_input):
                print(token, end='', flush=True)
            print()  # New line after response
        except Exception as e:
            print(f"\nError generating response: {e}")
            print("Please try again or check the model configuration.")

if __name__ == "__main__":
    main()



#What To do

#this just addon.
#download dataset from huggingface with data_model.json and history of download
#create class to finetune model with its compatible dataset Multiple time and can be chain function

#main stuff
#create event loop for use input and model input (should recieve multiple input type data as sametime)
#create attention between event loop for filter unwanting data so runtime not interfere
#create embbeding and en-router-attention with de-router-attention (shared attention or embed)
#create function feed input from router to encoder_model or decoder model
#create function to display output by model output from router_attention
#####note output from model should be stream into input of model_input instead of user_input or its model input for inteferencing
#####the data should be on eventloop instead of model loop so crack that 1 bit llms