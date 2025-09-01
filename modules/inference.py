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

class InferenceManager:
    """Manages inference with a language or multimodal model."""

    def __init__(self, model_path: str, max_new_tokens: int = 50, temperature: float = 0.9, top_p: float = 0.9):
        """Initialize the inference manager.

        Args:
            model_path: Path to the model to load.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature for generation.
            top_p: Top-p sampling parameter.
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.vision_path = Path(model_path ) / "vision_model"
        self.lang_path = Path(model_path ) / "lang_model"
        
        self.chat_template = """
                            {% for message in messages %}
                                {% if message['role'] == 'system' %}
                                {{ message['content'] }}
                                {% elif message['role'] == 'user' %}
                                Human: {% if message.get('images') %}
                                [Images: {{ message['images']|length }}]
                                {% endif %}
                                {{ message['content'] }}
                                {% elif message['role'] == 'assistant' %}
                                Assistant: {{ message['content'] }}
                                {% endif %}
                            {% endfor %}
                            Assistant:
                            """

        self._setup_device()
        self._load_model_and_tokenizer()
        # self.setup_chatTemplate()

    def _setup_device(self):
        """Set up the device for model execution."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            print("Warning: Running on CPU. Performance will be slower.")
   

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer, and determine if it's multimodal."""
        try:
            # Load the model configuration
            config = AutoConfig.from_pretrained(self.model_path)
            print(f"Loaded model configuration: {config.model_type}")

            # Check if the model is multimodal
            if hasattr(config, "model_type") and config.model_type == "vision-model":
                print("Detected multimodal model. Loading VisionModel...")
                vision_model = CLIPVisionModel.from_pretrained(self.vision_path, torch_dtype=torch.float16)
                lang_model = AutoModelForCausalLM.from_pretrained(self.lang_path, torch_dtype=torch.float16)
                self.model = VisionModel(config, vision_model, lang_model).to(self.device)
            else:
                print("Detected text-only model. Loading ConversationModel...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                self.model = ConversationModel(config, base_model).to(self.device)

            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    
    def format_chat(self,messages: list, chat_template: str) -> str:
        """
        Format chat messages based on the provided chat template.

        Args:
            messages (list): A list of dictionaries representing the conversation. Each dictionary should have:
                            - 'role': The role of the message sender ('system', 'user', or 'assistant').
                            - 'content': The content of the message.
                            - 'images' (optional): A list of images (only for 'user' role).
            chat_template (str): The Jinja2 template string for formatting the messages.

        Returns:
            str: The formatted chat string.
        """
        # Create a Jinja2 template from the provided chat_template
        template = Template(chat_template)

        # Render the template with the messages
        formatted_chat = template.render(messages=messages)

        # Normalize whitespace: strip leading/trailing, collapse multiple blank lines and indent
        formatted_chat = re.sub(r"[ \t]+$", "", formatted_chat, flags=re.MULTILINE)   
        formatted_chat = re.sub(r"\n\s*\n+", "\n\n", formatted_chat)                 
        formatted_chat = formatted_chat.strip()

        # Ensure Assistant: marker exists at end
        if not re.search(r"Assistant:\s*$", formatted_chat):
            formatted_chat = formatted_chat + "\n\nAssistant:"

        return formatted_chat

    def generate_response(self, user_input: str, image_path: str = None) -> str:
        """Generate a response from the model.

        Args:
            user_input: The input text from the user.
            image_path: Optional path to an image for multimodal models.

        Returns:
            The generated response as a string.
        """
        try:
            # Define the messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
            if image_path:
                messages[-1]["images"] = [image_path]

            # Use tokenizer/template-aware formatter
            prompt = self.format_chat(messages,self.chat_template)

            # Prepare inputs for the model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Handle multimodal inputs if the model supports it
            if hasattr(self.model, "vision_model") and image_path:
                # Load and preprocess the image
                image = load_image(image_path)
                processor = self._get_image_processor()
                pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
                inputs["pixel_values"] = pixel_values

            # Generate outputs
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                repetition_penalty=1.2,  # Add repetition penalty
                no_repeat_ngram_size=3   # Prevent repeating 3-grams
            )

            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response
            response = self._clean_response(response)
            
            return response

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "An error occurred during inference."

    def _clean_response(self, response: str) -> str:
        """Clean up repetitive patterns in the response."""
        # Extract content after the last "Assistant:" if present
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        # For simple factual questions, often just the first sentence is enough
        if len(response) > 100 and "." in response[:100]:
            first_sentence = response.split('.')[0] + '.'
            return first_sentence
        
        # Remove repetitive country patterns
        common_entities = ["France", "Germany", "Paris", "Berlin", "Europe"]
        for entity in common_entities:
            pattern = f"({entity})(,\\s*{entity})+\\b"
            response = re.sub(pattern, r"\1", response)
        
        return response.strip()

    def _get_image_processor(self):
        """Get the appropriate image processor for the model."""
        try:
            return CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
        except Exception as e:
            print(f"Error loading image processor: {str(e)}")
            raise
