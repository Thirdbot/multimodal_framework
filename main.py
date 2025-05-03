import os
import gc
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml
from accelerate.utils import set_seed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig
)

from modules.huggingface_login import HuggingFaceLogin
from modules.finetuning_model import FinetuneModel

# Set environment variables for optimization

# Define paths
WORKSPACE_DIR = Path(__file__).parent.absolute()
MODEL_DIR = WORKSPACE_DIR / "models" / "Text-Text-generation"
OFFLOAD_DIR = WORKSPACE_DIR / "offload"
OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)






#What To do

#this just addon.
#download dataset from huggingface with data_model.json and history of download
#create class to finetune model with its compatible dataset Multiple time and can be chain function

#main stuff
#create event loop for use input and model input (should recieve multiple input type data as sametime)
#create attention between event loop for filter unwanting data so runtime not interfere
#create embbeding and en-router-attention with de-router-attention (shared attention or embed)
#create function feed input from router to encoder_model or decoder model
#create function to display output by model output from router_attention
#####note output from model should be stream into input of model_input instead of user_input or its model input for inteferencing
#####the data should be on eventloop instead of model loop so crack that 1 bit llms