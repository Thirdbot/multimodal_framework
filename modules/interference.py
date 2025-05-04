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

init(autoreset=True)

class Inference:
    def __init__(self):
        # Define paths
        self.WORKSPACE_DIR = Path(__file__).parent.parent.absolute()
        self.MODEL_DIR = self.WORKSPACE_DIR / "models" / "Text-Text-generation"
        self.OFFLOAD_DIR = self.WORKSPACE_DIR / "offload"
        
        # Create directories if they don't exist
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Set memory optimization environment variables
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()
        
        # Initialize ChatTemplate
        self.chat_template = None

    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            print(f"{Fore.CYAN}Loading fine-tuned model from {self.MODEL_DIR}{Style.RESET_ALL}")
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

            # Load model
            model = AutoPeftModelForCausalLM.from_pretrained(
                self.MODEL_DIR,
                device_map=self.device_map,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                offload_folder=str(self.OFFLOAD_DIR),
                trust_remote_code=True
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_DIR,
                trust_remote_code=True,
                padding_side="right"
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.pad_token_id
                
            # Initialize ChatTemplate with the loaded tokenizer
            self.chat_template = ChatTemplate(tokenizer=tokenizer)

            return model, tokenizer

        except Exception as e:
            print(f"{Fore.RED}Error loading fine-tuned model: {str(e)}{Style.RESET_ALL}")
            return None, None

    def generate_text(self, prompt, max_length=100, temperature=0.7, top_p=0.9, is_chat=False):
        """Generate text using the fine-tuned model"""
        try:
            model, tokenizer = self.load_model()
            if model is None or tokenizer is None:
                raise ValueError("Failed to load model or tokenizer")
                
            if self.chat_template is None:
                raise ValueError("ChatTemplate not initialized")

            # Format the input based on whether it's a chat or regular text
            if is_chat:
                if isinstance(prompt, str):
                    # Convert string to chat format
                    prompt = [{"role": "user", "content": prompt}]
                formatted_prompt = self.chat_template.format_conversation(prompt)
            else:
                formatted_prompt = prompt

            # Create text generation pipeline
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=self.device_map
            )

            # Generate text
            output = generator(
                formatted_prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )

            generated_text = output[0]['generated_text']
            
            if is_chat:
                # Extract only the assistant's response
                generated_text = generated_text[len(formatted_prompt):].strip()
                
            print(f"{Fore.GREEN}Generated text: {generated_text}{Style.RESET_ALL}")
            return generated_text

        except Exception as e:
            print(f"{Fore.RED}Error generating text: {str(e)}{Style.RESET_ALL}")
            return None

    def chat(self, messages, max_length=100, temperature=0.7, top_p=0.9):
        """Generate a chat response"""
        return self.generate_text(
            messages,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            is_chat=True
        )

    def __call__(self, prompt, **kwargs):
        """Make the class callable for easy text generation"""
        return self.generate_text(prompt, **kwargs)
