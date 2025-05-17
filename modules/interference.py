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
init(autoreset=True)

class ConversationManager:
    def __init__(self, model_name: str = None, max_length: int = 100, temperature: float = 0.7, top_p: float = 0.9):
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = 50  # Maximum number of new tokens to generate
        
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
        self.max_memory_length = 10
        
        # Initialize message structure
        self.messages = {
            "system": {
                "role": "system",
                "content": [{"text": "You are Rick from Rick and Morty. Respond in character."}]
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
            else:
                model_path = self.MODEL_DIR / model_name
            
            # Load model
            self.model = AutoPeftModelForCausalLM.from_pretrained(
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
                tokenizer=self.tokenizer,
            )
            
            print(f"{Fore.GREEN}Successfully loaded model and tokenizer from {model_path}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error loading model: {str(e)}{Style.RESET_ALL}")
            raise
    
    def format_messages(self) -> str:
        """Format messages into a prompt string using our custom template"""
        # Build conversation string
        conversation = []
        
        # Add system message
        if self.messages["system"]["content"][0]["text"]:
            conversation.append(self.messages["system"]["content"][0]["text"])
        
        # Add user message if not empty
        if self.messages["user"]["content"][0]["text"]:
            conversation.append(f"Human: {self.messages['user']['content'][0]['text']}")
        
        # Add assistant message if not empty
        if self.messages["assistant"]["content"][0]["text"]:
            conversation.append(f"Assistant: {self.messages['assistant']['content'][0]['text']}")
        
        # Join with newlines and add final prompt
        formatted = "\n\n".join(conversation)
        if formatted:
            formatted += "\n\nAssistant:"
        
        return formatted
    
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
                "num_beams": 1,  # Disable beam search
                "early_stopping": False  # Disable early stopping since we're not using beam search
            }
            
            # Tokenize with truncation
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length - self.max_new_tokens
            ).to(self.device_map)
            
            # Generate response
            output = self.generator(
                prompt,
                **generation_config
            )
            
            response = output[0]['generated_text']
            # Remove the input prompt from the response
            response = response[len(prompt):].strip()
            
            # Clean up response
            if response.startswith("Assistant:"):
                response = response[len("Assistant:"):].strip()
            
            # Remove any repeated phrases
            words = response.split()
            cleaned_words = []
            for i, word in enumerate(words):
                if i > 0 and word == words[i-1]:
                    continue
                cleaned_words.append(word)
            response = " ".join(cleaned_words)
            
            return response
            
        except Exception as e:
            print(f"{Fore.RED}Error generating response: {str(e)}{Style.RESET_ALL}")
            return None
    
    def chat(self, user_input: str) -> str:
        """Process a user input and generate a response"""
        try:
            # Update user message
            self.messages["user"]["content"][0]["text"] = user_input
            
            # Format the conversation
            formatted_prompt = self.format_messages()
            
            # Generate response
            response = self.generate_response(formatted_prompt)
            
            if response:
                # Check for repetition
                if len(response.split()) < 3:  # If response is too short
                    response = self.generate_response(formatted_prompt)  # Try again
                
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
                "content": [{"text": "You are Rick from Rick and Morty. Respond in character."}]
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

def main():
    # Example usage
    try:
        # Initialize conversation manager (will use latest finetuned model)
        manager = ConversationManager(
            max_length=100,
            temperature=0.7,
            top_p=0.9
        )
        
        print(f"{Fore.CYAN}Starting conversation. Type 'quit' to exit.{Style.RESET_ALL}")
        
        while True:
            # Get user input
            user_input = input(f"{Fore.YELLOW}You: {Style.RESET_ALL}")
            if user_input.lower() == 'quit':
                break
            
            # Generate response
            response = manager.chat(user_input)
            if response:
                print(f"{Fore.GREEN}Assistant: {response}{Style.RESET_ALL}")
        
        # Print conversation memory
        print(f"\n{Fore.CYAN}Conversation Memory:{Style.RESET_ALL}")
        for exchange in manager.get_memory():
            print(f"User: {exchange['user']}")
            print(f"Assistant: {exchange['assistant']}\n")
            
    except Exception as e:
        print(f"{Fore.RED}Error in main: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
