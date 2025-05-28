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
from typing import List, Dict, Optional, Union
import re

# from langchain.llms import HuggingFacePipeline
# from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

init(autoreset=True)

class ConversationManager:
    def __init__(self, model_name: str = None, max_length: int = 100, temperature: float = 0.7, top_p: float = 0.9):
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = 100000  # Maximum number of new tokens to generate
        
        # Set up paths
        self.WORKSPACE_DIR = Path(__file__).parent.parent.absolute()
        self.MODEL_DIR = self.WORKSPACE_DIR / "models" / "text-generation"
        self.OFFLOAD_DIR = self.WORKSPACE_DIR / "offload"
        
        # Create directories
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Initialize conversation state
        self.memory = []
        self.max_memory_length = 10000
        
        # Initialize message structure
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
        
        # Load model and tokenizer
        self._load_model(model_name)
        
    def _load_model(self, model_name: Optional[str] = None):
        """Load the model and tokenizer with proper configuration"""
        try:
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            # If no model name provided, use the latest finetuned model
            if model_name is None:
                # Get the latest model directory
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

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map=self.device_map,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                offload_folder=str(self.OFFLOAD_DIR),
                trust_remote_code=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            # Initialize chat template
            self.chat_template = ChatTemplate(
                chainpipe=self.model,
                tokenizer=self.tokenizer
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            print(f"{Fore.GREEN}Successfully loaded model and tokenizer from {model_path}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error loading model: {str(e)}{Style.RESET_ALL}")
            raise
    
    def format_messages(self, history=None) -> str:
        """Format messages into a prompt string using our custom template"""
        prompt = ""
        
        # Add system message if present
        if self.messages["system"]["content"][0]["text"]:
            prompt += f"system: {self.messages['system']['content'][0]['text']}\n"
    
        for turn in history:
            if isinstance(turn, dict):
                # Handle different message formats
                if "user" in turn and turn["user"]:
                    prompt += f"Human: {turn['user']}\n"
                elif "assistant" in turn and turn["assistant"]:
                    prompt += f"Assistant: {turn['assistant']}\n"
                elif "role" in turn and "content" in turn:
                    if turn["role"] == "system":
                        prompt += f"system: {turn['content']}\n"
                    elif turn["role"] == "user":
                        prompt += f"Human: {turn['content']}\n"
                    elif turn["role"] == "assistant":
                        prompt += f"Assistant: {turn['content']}\n"
    
        # Add current user message if not empty
        if self.messages["user"]["content"][0]["text"]:
            prompt += f"Human: {self.messages['user']['content'][0]['text']}\n"
        
        # Add current assistant message if not empty
        if self.messages["assistant"]["content"][0]["text"]:
            prompt += f"Assistant: {self.messages['assistant']['content'][0]['text']}\n"
        
        # Add final prompt if there's any content
        if prompt:
            prompt += "Assistant:"
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response using the model"""
        try:
            # Configure generation parameters
            generation_config = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_return_sequences": 1,
                "pad_token_id": self.tokenizer.pad_token_id,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
                "num_beams": 1,
                "early_stopping": False
            }
            
            # Generate response
            output = self.generator(
                prompt,
                **generation_config
            )
            
            response = output[0]['generated_text']
            # Remove the input prompt from the response
            response = response[len(prompt):].strip()
            
            # Get the first complete response (up to the first role marker or newline)
            response = response.split('\n')[0].split('system:')[0].split('Human:')[0].split('Assistant:')[0].strip()
            
            return response
            
        except Exception as e:
            print(f"{Fore.RED}Error generating response: {str(e)}{Style.RESET_ALL}")
            return None
    
    def chat(self, user_input: str) -> str:
        """Process a user input and generate a response"""
        try:
            # Update user message
            self.messages["user"]["content"][0]["text"] = user_input
            self.memory.append(self.messages)
            # Format the conversation with memory
            formatted_prompt = self.format_messages(self.memory)
            
            # Generate response
            response = self.generate_response(formatted_prompt)
            
            if response:
                # Update assistant message
                self.messages["assistant"]["content"][0]["text"] = response
                
                # Update memory
                self.memory.append({
                    "user": user_input,
                    "assistant": response
                })
                
                # Trim memory if needed
                if len(self.memory) > self.max_memory_length:
                    self.memory = self.memory[-self.max_memory_length:]
                
                return response
            
            return None
            
        except Exception as e:
            print(f"{Fore.RED}Error in chat: {str(e)}{Style.RESET_ALL}")
            return None
    
    def get_memory(self) -> List[Dict]:
        """Get the conversation memory"""
        return self.memory
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory = []
        # Reset message structure
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
