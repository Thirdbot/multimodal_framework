# --base model using transformers pipeline

# --base model architecture change

# --fine-tune and merge with base-model through finetuning_model.py with custom multimodal embedding

#  -- use a model distillation method to distill the base-model into a smaller model

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")