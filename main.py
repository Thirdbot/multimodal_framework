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
from colorama import Fore, Style, init
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
from modules.inference import ModelInference
from modules.interference import Inference
from modules.chatTemplate import ChatTemplate
import argparse

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
        self.HomePath = Path(__file__).parent.absolute()
        
        self.DataModelFolder = f"{self.HomePath}/DataModel_config"
        Path(self.DataModelFolder).mkdir(parents=True, exist_ok=True)
        self.temporal_file_path = f'{self.DataModelFolder}/data_model.json'
        Path(self.temporal_file_path).touch(exist_ok=True)
        
        self.datafile = 'installed.json'
        self.model_data_json_path = f"{self.DataModelFolder}/{self.datafile}"
        Path(self.model_data_json_path).touch(exist_ok=True)
        
        self.dataset_handler = DatasetHandler()
        self.data_loader = DataLoader()
        self.finetune_model = FinetuneModel(model_data_json_path=self.model_data_json_path)
        
        self.model_data_params = {
            "model_name":["beatajackowska/DialoGPT-RickBot"],
            "datasets_name":["theneuralmaze/rick-and-morty-transcripts-sharegpt"]
        }
        
        # Initialize ChatTemplate
        self.chat_template = None

    def runtrain(self):
        model = None
        dataset = None
        failed_models = None
        try:
            #fetch api data
            self.dataset_handler.handle_data(self.temporal_file_path,**self.model_data_params)
            #load data from api
            failed_models = self.data_loader.run(self.model_data_params)
            self.list_model_data = self.finetune_model.generate_model_data()
            #finetune model
            model,dataset = self.finetune_model.run_finetune(self.list_model_data)


           
        except Exception as e:
            report = Report()
            if model and dataset is not None:
                report.store_problem(model=model,dataset=dataset)
            else:
                # report.store_problem(model=model,dataset=dataset)
                print(Fore.RED + f"Error happen: {e} but no reported." + Style.RESET_ALL)
            if failed_models is not None:
                for element in failed_models:
                    report.store_problem(model=element['model'],dataset=element['datasets'])
            

if __name__ == "__main__":
    main = Main()
    main.runtrain()
      # Example usage
    inference = ModelInference()
    
    model_name = "DialoGPT-RickBot"
    if inference.load_model(model_name):
        # Get model info
        info = inference.get_model_info()
        print(f"{Fore.CYAN}Model Info:{Style.RESET_ALL}")
        print(json.dumps(info, indent=2))
        
        # Generate text
        prompt = "who are you"
        results = inference.generate(prompt)
        if results:
            print(f"{Fore.CYAN}Generated Text:{Style.RESET_ALL}")
            for i, result in enumerate(results):
                print(f"{Fore.GREEN}Result {i+1}:{Style.RESET_ALL}")
                print(result)
# 
# What To do

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