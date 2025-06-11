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
import json

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
        self.HISTORY_DIR = self.WORKSPACE_DIR / "conversation_history"
        
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    
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
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_cache=True
        )
        
        # Optimize model for inference
        self.model.eval()
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            padding_side="right",
            use_fast=True
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
        """Format messages into a prompt string with better context handling."""
        prompt = []
        
        # Add system message first
        if self.messages["system"]["content"][0]["text"]:
            prompt.append(f"system: {self.messages['system']['content'][0]['text']}")
        
        # Add conversation history with clear turn markers
        if history:
            for turn in history:
                if isinstance(turn, dict):
                    if "user" in turn and turn["user"]:
                        prompt.append(f"Human: {turn['user']}")
                    elif "assistant" in turn and turn["assistant"]:
                        prompt.append(f"Assistant: {turn['assistant']}")
        
        # Add current user message
        if self.messages["user"]["content"][0]["text"]:
            prompt.append(f"Human: {self.messages['user']['content'][0]['text']}")
        
        # Add assistant marker for response
        prompt.append("Assistant:")
        
        return "\n".join(prompt)
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate a response using the model with improved context handling."""
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
                "repetition_penalty": 1.2,  # Increased to reduce repetition
                "no_repeat_ngram_size": 3,  # Increased to improve coherence
                "num_beams": 1,
                "early_stopping": True,
                "use_cache": True,
                "return_dict_in_generate": True,
                "output_scores": False,
                "output_hidden_states": False,
                "min_length": 10,  # Ensure responses aren't too short
                "length_penalty": 1.0,  # Balance between short and long responses
            }
            
            with torch.inference_mode():
                with torch.cuda.amp.autocast():
                    output = self.generator(prompt, **generation_config)
                    response = output[0]['generated_text'][len(prompt):].strip()
                    
                    # Clean up response
                    response = response.split('\n')[0]  # Take first line
                    response = response.split('Human:')[0]  # Remove any Human: markers
                    response = response.split('Assistant:')[0]  # Remove any Assistant: markers
                    response = response.split('system:')[0]  # Remove any system: markers
                    response = response.strip()
                    
                    # Ensure response is not empty
                    if not response:
                        response = "I apologize, but I need more context to provide a meaningful response."
                    
                    # Cache the result
                    self.cache[prompt] = response
                    return response
            
        except Exception as e:
            print(f"{Fore.RED}Error generating response: {str(e)}{Style.RESET_ALL}")
            return None
    
    def _load_latest_memory(self) -> List[Dict]:
        """Load the latest conversation memory from files.
        
        Returns:
            List of conversation turns or empty list if no history found
        """
        try:
            # Get all history files
            history_files = list(self.HISTORY_DIR.glob("chat_*.json"))
            if not history_files:
                return []
            
            # Get the latest file
            latest_file = max(history_files, key=lambda x: x.stat().st_mtime)
            
            # Load the conversation history
            with open(latest_file, 'r') as f:
                history = json.load(f)
                print(f"{Fore.GREEN}Loaded conversation history from {latest_file}{Style.RESET_ALL}")
                return history
                
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not load conversation history: {str(e)}{Style.RESET_ALL}")
            return []

    def chat(self, user_input: str) -> Optional[str]:
        """Process a user input and generate a response.
        
        Args:
            user_input: User's input message
            
        Returns:
            Generated response or None if generation fails
        """
        try:
            # Load memory from file if empty
            if len(self.memory) == 0:
                self.memory = self._load_latest_memory()
                if self.memory:
                    print(f"{Fore.CYAN}Using previous conversation context with {len(self.memory)} turns{Style.RESET_ALL}")
            
            self.messages["user"]["content"][0]["text"] = user_input
            
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
