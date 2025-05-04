import os
import json
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig
)
from peft import PeftModel, PeftConfig
from huggingface_hub import HfApi
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

class ModelInference:
    def __init__(self, workspace_dir=None):
        # Set up paths
        if workspace_dir is None:
            self.WORKSPACE_DIR = Path(__file__).parent.parent.absolute()
        else:
            self.WORKSPACE_DIR = Path(workspace_dir)
            
        self.MODEL_DIR = self.WORKSPACE_DIR / "models"
        self.OFFLOAD_DIR = self.WORKSPACE_DIR / "offload"
        
        # Create offload directory if needed
        self.OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Set memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.cuda.empty_cache()
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.model_info = None
        self.model_path = None

    def get_model_task(self, model_name):
        """Get the task type for a model from Hugging Face"""
        try:
            api = HfApi()
            models = api.list_models(search=model_name)
            for model in models:
                if model.id.startswith(model_name):
                    return model.pipeline_tag
            return "text-generation"
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not determine model task, using default: {str(e)}{Style.RESET_ALL}")
            return "text-generation"

    def load_model(self, model_name):
        """Load a model from the local directory"""
        try:
            # Get model task and path
            model_task = self.get_model_task(model_name)
            model_path = self.MODEL_DIR / model_task / model_name
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model directory not found: {model_path}")
            
            # Load model info
            model_info_path = model_path / "model_info.json"
            if not model_info_path.exists():
                raise FileNotFoundError(f"Model info file not found: {model_info_path}")
            
            with open(model_info_path, "r") as f:
                self.model_info = json.load(f)
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                padding_side="right",
                truncation_side="right"
            )
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_info["base_model"],
                quantization_config=quantization_config,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                offload_folder=str(self.OFFLOAD_DIR),
                offload_state_dict=True,
                max_memory={0: "40GB"}
            )
            
            # Load PEFT model if finetuned
            if self.model_info.get("finetuned", False):
                self.model = PeftModel.from_pretrained(
                    base_model,
                    str(model_path),
                    device_map=self.device
                )
            else:
                self.model = base_model
            
            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            self.model_path = model_path
            print(f"{Fore.GREEN}Model loaded successfully from {model_path}{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error loading model: {str(e)}{Style.RESET_ALL}")
            return False

    def generate(self, prompt, max_length=512, temperature=0.7, top_p=0.9, num_return_sequences=1):
        """Generate text based on the prompt"""
        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model and tokenizer must be loaded first")
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode and return results
            results = []
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                results.append(text)
            
            return results
            
        except Exception as e:
            print(f"{Fore.RED}Error during generation: {str(e)}{Style.RESET_ALL}")
            return None

    def get_model_info(self):
        """Return information about the loaded model"""
        if self.model_info is None:
            return None
        return {
            "model_id": self.model_info["model_id"],
            "model_task": self.model_info["model_task"],
            "base_model": self.model_info["base_model"],
            "finetuned": self.model_info["finetuned"],
            "quantization": self.model_info["quantization"],
            "lora_config": self.model_info["lora_config"]
        }

