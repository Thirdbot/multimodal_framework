import os
import torch
from pathlib import Path
from colorama import Fore, Style, init
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import AutoPeftModelForCausalLM
from modules.chatTemplate import ChatTemplate
from modules.chainpipe import Chainpipe
from typing import List, Dict, Optional, Union, Any
import re

# from langchain.llms import HuggingFacePipeline
# from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

init(autoreset=True)

class ConversationManager:
    """Manages conversation with a language model."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """Initialize the conversation manager.
        
        Args:
            model_name: Name or path of the model to use
            max_length: Maximum length of generated responses
            temperature: Sampling temperature for generation
            top_p: Top-p sampling parameter
        """
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = 512  # Reduced from 100000 for faster inference
        
        self._setup_paths()
        self._setup_device()
        self._initialize_state()
        self._load_model(model_name)
        
        # Initialize cache for faster repeated queries
        self.cache = {}
    
    def _setup_paths(self):
        """Set up directory paths."""
        self.WORKSPACE_DIR = Path(__file__).parent.parent.absolute()
        self.MODEL_DIR = self.WORKSPACE_DIR / "models" / "text-generation"
        self.OFFLOAD_DIR = self.WORKSPACE_DIR / "offload"
        
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    def _setup_device(self):
        """Set up the device for model execution with optimized GPU settings."""
        if torch.cuda.is_available():
            self.device_map = "cuda:0"
            # Clear GPU cache
            torch.cuda.empty_cache()
            # Set memory growth to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.8)
            # Enable TF32 for faster computation on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Enable cudnn benchmarking for faster inference
            torch.backends.cudnn.benchmark = True
        else:
            self.device_map = "cpu"
            print(f"{Fore.YELLOW}Warning: Running on CPU. Performance will be significantly slower.{Style.RESET_ALL}")
    
    def _initialize_state(self):
        """Initialize conversation state and message structure."""
        self.memory = []
        self.max_memory_length = 10000
        
        self.messages = {
            "system": {
                "role": "system",
                "content": [{"text": "You are a helpful AI assistant."}]
            },
            "user": {
                "role": "user",
                "content": [{"text": ""}]
            },
            "assistant": {
                "role": "assistant",
                "content": [{"text": ""}]
            }
        }
    
    def _load_model(self, model_name: Optional[str] = None):
        """Load the model and tokenizer.
        
        Args:
            model_name: Name or path of the model to load
        """
        try:
            model_path = self._get_model_path(model_name)
            self._load_model_and_tokenizer(model_path)
            self._setup_chat_template()
            self._setup_generator()
            
            print(f"{Fore.GREEN}Successfully loaded model and tokenizer from {model_path}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error loading model: {str(e)}{Style.RESET_ALL}")
            raise
    
    def _get_model_path(self, model_name: Optional[str]) -> Path:
        """Get the path to the model.
        
        Args:
            model_name: Name or path of the model
            
        Returns:
            Path to the model
        """
        if model_name is None:
            model_dirs = list(self.MODEL_DIR.glob("*"))
            if not model_dirs:
                raise ValueError("No finetuned models found in the models directory")
            model_path = max(model_dirs, key=lambda x: x.stat().st_mtime)
            print(f"{Fore.CYAN}Using latest model from: {model_path}{Style.RESET_ALL}")
        elif (self.MODEL_DIR / model_name).exists():
            print(f"{Fore.CYAN}Using model from: {self.MODEL_DIR / model_name}{Style.RESET_ALL}")
            model_path = self.MODEL_DIR / model_name
        else:
            print(f"{Fore.CYAN}Using model from: {model_name}{Style.RESET_ALL}")
            model_path = model_name
            
        return model_path
    
    def _load_model_and_tokenizer(self, model_path: Union[str, Path]):
        """Load the model and tokenizer with optimized settings for GPU."""
        # Enable model optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map=self.device_map,
            torch_dtype=torch.float16,  # Use float16 for better memory efficiency
            low_cpu_mem_usage=True,
            offload_folder=str(self.OFFLOAD_DIR),
            trust_remote_code=True,
            use_cache=True  # Enable KV cache
        )
        
        # Optimize model for inference
        self.model.eval()  # Set to evaluation mode
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        
        # Enable model optimizations
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            padding_side="right",
            use_fast=True  # Use fast tokenizer
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
    
    def _setup_chat_template(self):
        """Set up the chat template."""
        self.chat_template = ChatTemplate(tokenizer=self.tokenizer)
    
    def _setup_generator(self):
        """Set up the text generation pipeline."""
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
    
    def format_messages(self, history: Optional[List[Dict]] = None) -> str:
        """Format messages into a prompt string.
        
        Args:
            history: Conversation history
            
        Returns:
            Formatted prompt string
        """
        prompt = []
        
        if self.messages["system"]["content"][0]["text"]:
            prompt.append(f"system: {self.messages['system']['content'][0]['text']}")
        
        if history:
            for turn in history:
                if isinstance(turn, dict):
                    if "user" in turn and turn["user"]:
                        prompt.append(f"Human: {turn['user']}")
                    elif "assistant" in turn and turn["assistant"]:
                        prompt.append(f"Assistant: {turn['assistant']}")
                    elif "role" in turn and "content" in turn:
                        role_prefix = "system" if turn["role"] == "system" else "Human" if turn["role"] == "user" else "Assistant"
                        prompt.append(f"{role_prefix}: {turn['content']}")
        
        if self.messages["user"]["content"][0]["text"]:
            prompt.append(f"Human: {self.messages['user']['content'][0]['text']}")
        
        if self.messages["assistant"]["content"][0]["text"]:
            prompt.append(f"Assistant: {self.messages['assistant']['content'][0]['text']}")
        
        prompt.append("Assistant:")
        
        return "\n".join(prompt)
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate a response using the model with optimized settings."""
        try:
            # Check cache first
            if prompt in self.cache:
                return self.cache[prompt]
            
            generation_config = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_return_sequences": 1,
                "pad_token_id": self.tokenizer.pad_token_id,
                "repetition_penalty": 1.1,  # Reduced from 1.2
                "no_repeat_ngram_size": 2,  # Reduced from 3
                "num_beams": 1,
                "early_stopping": True,  # Enable early stopping
                "use_cache": True,  # Enable KV cache
                "return_dict_in_generate": True,
                "output_scores": False,  # Disable score computation for speed
                "output_hidden_states": False,  # Disable hidden states for speed
            }
            
            with torch.inference_mode():  # Use inference mode for faster computation
                with torch.cuda.amp.autocast():  # Enable automatic mixed precision
                    output = self.generator(prompt, **generation_config)
                    response = output[0]['generated_text'][len(prompt):].strip()
                    
                    # Extract first complete response
                    response = response.split('\n')[0].split('system:')[0].split('Human:')[0].split('Assistant:')[0].strip()
                    
                    # Cache the result
                    self.cache[prompt] = response
                    return response
            
        except Exception as e:
            print(f"{Fore.RED}Error generating response: {str(e)}{Style.RESET_ALL}")
            return None
    
    def chat(self, user_input: str) -> Optional[str]:
        """Process a user input and generate a response.
        
        Args:
            user_input: User's input message
            
        Returns:
            Generated response or None if generation fails
        """
        try:
            self.messages["user"]["content"][0]["text"] = user_input
            # self.memory.append(self.messages)
            
            formatted_prompt = self.format_messages(self.memory)
            response = self.generate_response(formatted_prompt)
            
            if response:
                self._update_conversation_state(user_input, response)
                return response
            
            return None
            
        except Exception as e:
            print(f"{Fore.RED}Error in chat: {str(e)}{Style.RESET_ALL}")
            return None
    
    def _update_conversation_state(self, user_input: str, response: str):
        """Update the conversation state with new messages.
        
        Args:
            user_input: User's input message
            response: Generated response
        """
        self.messages["assistant"]["content"][0]["text"] = response
        self.memory.append({
            "user": user_input,
            "assistant": response
        })
        
        if len(self.memory) > self.max_memory_length:
            self.memory = self.memory[-self.max_memory_length:]
    
    def get_memory(self) -> List[Dict]:
        """Get the conversation memory.
        
        Returns:
            List of conversation turns
        """
        return self.memory
    
    def clear_memory(self):
        """Clear the conversation memory and reset message structure."""
        self.memory = []
        self.messages = {
            "system": {
                "role": "system",
                "content": [{"text": "You are a helpful AI assistant."}]
            },
            "user": {
                "role": "user",
                "content": [{"text": ""}]
            },
            "assistant": {
                "role": "assistant",
                "content": [{"text": ""}]
            }
        }
