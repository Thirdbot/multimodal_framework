import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#for create base model to be use for finetuning and customize architecture
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


#goal is to make a tokenizer and a model and save it to the models directory
#
class CreateModel:
    def __init__(self, model_name,model_path):
        self.model_name = model_name
        self.model_path = model_path
