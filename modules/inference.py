"""
inference.py – Run text (and optionally vision) inference on a saved model.

Usage (CLI):
    python -m modules.inference --model checkpoints/text-generation/MyModel --prompt "Hello"

Usage (GUI / code):
    manager = InferenceManager("path/to/model")
    response = manager.generate_response("What is this?")
"""

import re
from io import BytesIO
from pathlib import Path

import requests
import torch
from colorama import Fore, Style
from jinja2 import Template
from PIL import Image
from transformers import AutoImageProcessor

from modules.ModelUtils import load_saved_model
from modules.variable import Variable

# Default generation parameters
DEFAULT_MAX_NEW_TOKENS     = 1000
DEFAULT_TEMPERATURE        = 0.7
DEFAULT_TOP_P              = 0.9
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_NO_REPEAT_NGRAM    = 3

# Vision processor used for image inputs
VISION_PROCESSOR_NAME = "microsoft/resnet-50"


class InferenceManager:
    """
    Load a fine-tuned model and generate responses.

    Supports text-only and vision (image + text) models.
    The chat template stored in the tokenizer is used to format prompts.
    """

    def __init__(self, model_path):
        """
        model_path – path to the saved model directory (str or Path).
        """
        self.model_path   = Path(model_path)
        self.variable     = Variable()
        self.chat_template = None

        self._setup_device()
        self._load_model_and_tokenizer()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _setup_device(self):
        """Select CUDA if available, otherwise CPU."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True
            torch.backends.cudnn.benchmark        = True
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"{Fore.CYAN}GPU memory allocated: {allocated:.2f} GB{Style.RESET_ALL}")
        else:
            print("Warning: running on CPU – inference will be slow.")

    def _load_model_and_tokenizer(self):
        """Load model + tokenizer and prepare the image processor."""
        self.model, self.tokenizer = load_saved_model(self.model_path)
        self.chat_template         = self.tokenizer.chat_template

        # Image processor for vision inputs
        self.vision_processor = AutoImageProcessor.from_pretrained(
            VISION_PROCESSOR_NAME, use_fast=True
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_response(self, user_input, image_path=None):
        """
        Generate a text response for user_input.

        user_input  – the user's text query (str)
        image_path  – optional URL or local path to an image (str)

        Returns the decoded response string.
        """
        try:
            messages = self._build_messages(user_input)
            prompt   = self._format_prompt(messages)
            inputs   = self._prepare_inputs(prompt, image_path)
            outputs  = self._run_generation(inputs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"generate_response error: {e}")
            return "An error occurred during inference."

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_messages(self, user_input):
        """Build the messages list for the chat template."""
        return [
            {"role": "system",  "content": "You are a helpful assistant."},
            {"role": "user",    "content": user_input},
        ]

    def _format_prompt(self, messages):
        """Render messages through the stored Jinja chat template."""
        template = Template(self.chat_template)
        formatted = template.render(messages=messages)

        # Clean up trailing whitespace and collapse blank lines
        formatted = re.sub(r"[ \t]+$", "", formatted, flags=re.MULTILINE)
        formatted = re.sub(r"\n\s*\n+", "\n\n", formatted)
        formatted = formatted.strip()

        # Ensure the prompt ends with the assistant turn marker
        if not re.search(r"<\|im_start\|>assistant\s*$", formatted):
            formatted += "\n<|im_start|>assistant"

        return formatted

    def _prepare_inputs(self, prompt, image_path):
        """Tokenize the prompt and optionally add pixel values for vision."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Move masks/values to float32
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].float()

        # Add image if the model supports vision and an image was provided
        if hasattr(self.model, "vision_model") and image_path:
            image  = self._load_image(image_path)
            if image:
                pixel_values = self.vision_processor(
                    images=image, return_tensors="pt"
                )["pixel_values"].to(self.device)
                inputs["pixel_values"] = pixel_values

        return inputs

    def _run_generation(self, inputs):
        """Run model.generate with default sampling parameters."""
        return self.model.generate(
            input_ids      = inputs["input_ids"],
            attention_mask = inputs.get("attention_mask"),
            pixel_values   = inputs.get("pixel_values"),
            max_new_tokens = DEFAULT_MAX_NEW_TOKENS,
            temperature    = DEFAULT_TEMPERATURE,
            top_p          = DEFAULT_TOP_P,
            do_sample      = True,
            repetition_penalty  = DEFAULT_REPETITION_PENALTY,
            no_repeat_ngram_size= DEFAULT_NO_REPEAT_NGRAM,
        )

    def _load_image(self, image_path):
        """Load an image from a URL or local file path. Returns a PIL Image or None."""
        try:
            response = requests.get(image_path)
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            pass
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Image load error [{image_path}]: {e}")
            return None


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a saved model.")
    parser.add_argument("--model",  required=True, help="Path to the model directory")
    parser.add_argument("--prompt", required=True, help="User input prompt")
    parser.add_argument("--image",  default=None,  help="Optional image URL or path")
    args = parser.parse_args()

    manager  = InferenceManager(args.model)
    response = manager.generate_response(args.prompt, args.image)
    print(response)
