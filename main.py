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
        self.conversation = Conversation()
        self.conversation_file = "conversation_history.json"
        self.prompt_config = PromptConfig()
        
    def load_model(self, model_path):
        """Load a finetuned model using direct model loading"""
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        
        # Create offload directory if it doesn't exist
        offload_dir = Path("offload")
        offload_dir.mkdir(exist_ok=True)
        
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True
        )
        
        # Load tokenizer and model directly
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            trust_remote_code=True,
            offload_folder=str(offload_dir),
            offload_state_dict=True,
            max_memory={
                0: "8GB",  # GPU memory
                "cpu": "32GB"  # CPU memory
            },
            offload_buffers=True,
            low_cpu_mem_usage=True
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        return self.model
    
    def generate_response(self, user_input: str, max_new_tokens: int = 150, 
                         temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate a response using the loaded model with conversation history"""
        if self.model is None:
            raise RuntimeError("Model must be loaded first")
        
        # Add user message to conversation
        self.conversation.add_message({
            "role": "user",
            "content": user_input
        })
        
        # Get chat template format
        messages = self.conversation.get_chat_template()
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # Add assistant's response to conversation
        self.conversation.add_message({
            "role": "assistant",
            "content": response
        })
        
        # Save conversation after each exchange
        self.conversation.save_conversation(self.conversation_file)
        
        return response

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
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
    
    # Try to load previous conversation
    model_manager.conversation.load_conversation(model_manager.conversation_file)
    
    print("\nChat Interface")
    print("Available commands:")
    print("- 'quit': Exit the program")
    print("- 'clear': Start a new conversation")
    print("- 'history': View conversation history")
    print("- 'config': View/change prompt configurations")
    print("- 'save': Save current prompt configuration")
    print("-" * 50)
    
    # Interactive loop
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            model_manager.conversation = Conversation()
            print("\nConversation cleared. Starting fresh conversation.")
            continue
        elif user_input.lower() == 'history':
            print("\nConversation History:")
            print("-" * 50)
            print(model_manager.conversation.get_chat_template())
            print("-" * 50)
            continue
        elif user_input.lower() == 'config':
            print("\nAvailable Prompt Configurations:")
            configs = model_manager.prompt_config.get_available_configs()
            for config_type, options in configs.items():
                print(f"\n{config_type}:")
                for option in options:
                    current = " (current)" if option == model_manager.prompt_config.current_config[config_type] else ""
                    print(f"  - {option}{current}")
            
            change = input("\nChange configuration? (y/n): ").lower()
            if change == 'y':
                config_type = input("Enter configuration type (system/user/assistant/context): ")
                config_name = input("Enter configuration name: ")
                try:
                    model_manager.prompt_config.set_config(config_type, config_name)
                    print(f"Configuration updated: {config_type}={config_name}")
                except ValueError as e:
                    print(f"Error: {e}")
            continue
        elif user_input.lower() == 'save':
            model_manager.prompt_config.save_config(model_manager.prompt_config_file)
            print(f"Prompt configuration saved to {model_manager.prompt_config_file}")
            continue
        
        print("\nAssistant: ", end='', flush=True)
        response = model_manager.generate_response(
            user_input,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9
        )
        print(response)

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