from dataclasses import dataclass
from pathlib import Path
from io import BytesIO
import re

import torch
from colorama import Fore, Style
from jinja2 import Template
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    CLIPProcessor,
)
import requests

from modules.ModelUtils import VisionModelWrapper
from modules.variable import Variable

from peft import PeftModel


@dataclass
class InferenceConfig:
    """Configuration for inference manager."""
    max_new_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    vision_processor_name: str = "openai/clip-vit-large-patch14"
    use_fast_tokenizer: bool = True
    torch_dtype: str = "auto"
    device_override: str | None = None
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    use_tf32: bool = True


class InferenceManager:
    def __init__(self, model_path: str, config: InferenceConfig | None = None):
        self.config = config or InferenceConfig()
        self.model_path = Path(model_path)
        self.vision_path = self.model_path / "vision_model"
        self.lang_path = self.model_path / "lang_model"
        self.vision_adapter_path = self.model_path / "vision_adapter"
        self.variable = Variable()
        self.dtype = torch.float32 if self.config.torch_dtype == "auto" else getattr(torch, self.config.torch_dtype)
        self.chat_template = None
        
        self._setup_device()
        self._load_model_and_tokenizer()

    def _setup_device(self):
        device_str = self.config.device_override or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"{Fore.CYAN}GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB{Style.RESET_ALL}")
            if self.config.use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
        else:
            print("Warning: Running on CPU. Performance will be slower.")
            
    def _load_model_and_tokenizer(self):
        try:
            config = AutoConfig.from_pretrained(self.model_path)
            print(f"Loaded model configuration: {config.model_type}")

            if hasattr(config, "model_type") and config.model_type == "vision-model":
                print("Detected multimodal model. Loading VisionModel...")
                
                self.vision_processor = CLIPProcessor.from_pretrained(
                    self.config.vision_processor_name,
                    use_fast=self.config.use_fast_tokenizer
                )

                pefted_lang_model = AutoModelForCausalLM.from_pretrained(
                    self.lang_path,
                    torch_dtype=self.dtype,
                    device_map="auto"
                )
                pefted_lang_model = PeftModel.from_pretrained(pefted_lang_model, self.lang_path)
                pefted_lang_model = pefted_lang_model.to(self.device)

                self.model = VisionModelWrapper(config, lang_model=pefted_lang_model)
        
                adapter_state_dict = torch.load(
                    self.vision_adapter_path / "vision_adapter.pt",
                    map_location=self.device,
                    weights_only=True
                )
                self.model.vision_adapter.load_state_dict(adapter_state_dict)
                self.model.vision_adapter = self.model.vision_adapter.to(self.device).to(self.dtype)
                self.model.vision_model = self.model.vision_model.to(self.device).to(self.dtype)

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.lang_path,
                    use_fast=self.config.use_fast_tokenizer
                )
            else:
                print("Detected text-only model. Loading ConversationModel...")
                # AutoModelForCausalLM will automatically use registered ConversationModel
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=self.dtype
                ).to(self.device)
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    use_fast=self.config.use_fast_tokenizer
                )

            # Ensure chat_template is loaded
            if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
                self.chat_template = self.tokenizer.chat_template
            elif hasattr(config, "chat_template") and config.chat_template is not None:
                self.chat_template = config.chat_template
            else:
                raise ValueError("Chat template not found in tokenizer or config.")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    
    def format_chat(self, messages: list, chat_template: str) -> str:
        template = Template(chat_template)
        formatted_chat = template.render(messages=messages)
        
        # Normalize whitespace
        formatted_chat = re.sub(r"[ \t]+$", "", formatted_chat, flags=re.MULTILINE)
        formatted_chat = re.sub(r"\n\s*\n+", "\n\n", formatted_chat)
        formatted_chat = formatted_chat.strip()

        if not re.search(r"<|im_start|>assistant:s*$", formatted_chat):
            formatted_chat = formatted_chat + "\n<|im_start|>assistant"

        return formatted_chat

    def load_images_url(self, image_path: str):
        try:
            url_content = requests.get(image_path).content
            image = Image.open(BytesIO(url_content)).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image from {image_path}: {str(e)}")
            return None
    def generate_response(self, user_input: str, image_path: str = None) -> str:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer questions based on the image if provided."},
                {"role": "user", "content": user_input}
            ]

            prompt = self.format_chat(messages, self.chat_template)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].to(self.dtype)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.device).to(self.dtype)

            if hasattr(self.model, "vision_model") and image_path:
                image = self.load_images_url(image_path)
                pixel_values = self.vision_processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
                inputs["pixel_values"] = pixel_values

            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                repetition_penalty=self.config.repetition_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "An error occurred during inference."