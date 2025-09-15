import os
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    CLIPVisionModel,
    CLIPProcessor,
)
from modules.chatTemplate import ChatTemplate
from modules.createbasemodel import VisionModel, ConversationModel
from transformers.image_utils import load_image
from jinja2 import Template
import re
from modules.variable import Variable
from peft import PeftModel, PeftConfig

class InferenceManager:

    def __init__(self, model_path: str, max_new_tokens: int = 1000, temperature: float = 0.7, top_p: float = 0.9):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.vision_path = Path(model_path ) / "vision_model"
        self.vision_processor_path = Path(model_path ) / "vision_processor"
        self.lang_path = Path(model_path ) / "lang_model"
        self.variable = Variable()
        self.dtype = self.variable.DTYPE
        
        self.chat_template = None
        
        self._setup_device()
        self._load_model_and_tokenizer()
        # self.setup_chatTemplate()

    def _setup_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            print("Warning: Running on CPU. Performance will be slower.")
   

    def _load_model_and_tokenizer(self):
        try:
            # Load the model configuration
            config = AutoConfig.from_pretrained(self.model_path)
            print(f"Loaded model configuration: {config.model_type}")

            # Check if the model is multimodal
            if hasattr(config, "model_type") and config.model_type == "vision-model":
                print("Detected multimodal model. Loading VisionModel...")
                # self.vision_model = CLIPVisionModel.from_pretrained(self.vision_path, torch_dtype=self.dtype)
                self.vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
                self.lang_model = AutoModelForCausalLM.from_pretrained(self.lang_path, torch_dtype=self.dtype)
                self.model = VisionModel(config, self.lang_model).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.lang_path, use_fast=True)
            else:
                print("Detected text-only model. Loading ConversationModel...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=self.dtype,
                    device_map="auto"
                )
                self.model = ConversationModel(config, base_model).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)

            # Load the tokenizer
            
            # Ensure chat_template is loaded, fallback to config if not present
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

    
    def format_chat(self,messages: list, chat_template: str) -> str:
        # Create a Jinja2 template from the provided chat_template
        template = Template(chat_template)

        # Render the template with the messages
        formatted_chat = template.render(messages=messages)        
        # Normalize whitespace: strip leading/trailing, collapse multiple blank lines and indent
        formatted_chat = re.sub(r"[ \t]+$", "", formatted_chat, flags=re.MULTILINE)   
        formatted_chat = re.sub(r"\n\s*\n+", "\n\n", formatted_chat)                 
        formatted_chat = formatted_chat.strip()

        # # Ensure Assistant: marker exists at end
        if not re.search(r"<|im_start|>assistant:s*$", formatted_chat):
            formatted_chat = formatted_chat + "\n<|im_start|>assistant"

        return formatted_chat

    def generate_response(self, user_input: str, image_path: str = None) -> str:
        try:
            # Define the messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer questions based on the image if provided."},
                {"role": "user", "content": user_input}
            ]
            # If image is provided, add a special token to the user message
            if image_path:
                messages[-1]["content"] += f" <images>{image_path}</images>"

            # Use tokenizer/template-aware formatter
            prompt = self.format_chat(messages, self.chat_template)
            # print(f"Formatted Prompt:\n{prompt}\n{'-'*50}")
            # Prepare inputs for the model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # model_dtype = next(self.model.parameters()).dtype

            # For attention_mask and pixel_values (if present)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].to(self.dtype)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.device).to(self.dtype)

            # Handle multimodal inputs if the model supports it
            if hasattr(self.model, "vision_model") and image_path:
                image = load_image(image_path)
                pixel_values = self.vision_processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
                inputs["pixel_values"] = pixel_values

            # Generate outputs
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )

            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # response = self._clean_response(response)
            return response

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "An error occurred during inference."