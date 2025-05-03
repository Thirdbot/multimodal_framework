import os
import gc
import json
from datetime import datetime
from pathlib import Path
# Set environment variables before any other imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism

import torch
import yaml
from accelerate.utils import set_seed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig
)

from modules.defect import Report
from modules.DataDownload import DataLoader
from modules.DatasetHandler import Manager as DatasetHandler
from modules.finetuning_model import Manager as FinetuneModel

# Set PyTorch settings
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Define paths
WORKSPACE_DIR = Path(__file__).parent.absolute()
MODEL_DIR = WORKSPACE_DIR / "models" / "Text-Text-generation"
OFFLOAD_DIR = WORKSPACE_DIR / "offload"
OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
set_seed(42)

class Main:
    def __init__(self):
        self.dataset_handler = DatasetHandler()
        self.data_loader = DataLoader()
        self.finetune_model = FinetuneModel()
        self.HomePath = Path(__file__).parent.absolute()
        
        self.DataModelFolder = f"{self.HomePath}/DataModel_config"
        self.temporal_file_path = f'{self.DataModelFolder}/ data_model.json'
        Path(self.temporal_file_path).touch(exist_ok=True)
        
        self.datafile = 'installed.json'
        self.model_data_json_path = f"{self.DataModelFolder}/{self.datafile}"
        Path(self.model_data_json_path).touch(exist_ok=True)
        
        self.manager = FinetuneModel(self.model_data_json_path)
        self.list_model_data = self.manager.generate_model_data()

    def run(self):
        try:
            self.dataset_handler.handle_data(self.temporal_file_path)
            self.data_loader.run()
            model,dataset = self.manager.run_finetune(self.list_model_data)
        except Exception as e:
            report = Report()
            report.store_problem(model=model,dataset=dataset)

if __name__ == "__main__":
    main = Main()
    main.run()

#What To do

#this just addon.

#create class to finetune model with its compatible dataset Multiple time and can be chain function

#main stuff
#create event loop for use input and model input (should recieve multiple input type data as sametime)
#create attention between event loop for filter unwanting data so runtime not interfere
#create embbeding and en-router-attention with de-router-attention (shared attention or embed)
#create function feed input from router to encoder_model or decoder model
#create function to display output by model output from router_attention
#####note output from model should be stream into input of model_input instead of user_input or its model input for inteferencing
#####the data should be on eventloop instead of model loop so crack that 1 bit llms