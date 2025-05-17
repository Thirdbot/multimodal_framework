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
from modules.interference import ConversationManager
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
            "datasets_name":["facebook/PLM-Image-Auto"],
            "model_amount":1,
            "datasets_amount":2,
            # "datasets_name":["OpenAssistant/oasst2"],
            # "task":["image-to-text"],
            # "search":"image",
            # "modality":"image"
        }
        self.config = dict()
        
        # Initialize ChatTemplate
        self.chat_template = None
    def load_datas(self):
        #place holder for load data
        self.dataset_handler.handle_data(self.temporal_file_path,**self.model_data_params)
        #load data from api
        self.data_loader.run(self.model_data_params)
        self.config = self.data_loader.saved_config 
        print(f"{Fore.CYAN}Dataset Config:{Style.RESET_ALL} {self.config}")
        
        
        
    def runtrain(self):
        model = None
        dataset = None
        failed_models = None
        try:
            # #fetch api data
            # self.dataset_handler.handle_data(self.temporal_file_path,**self.model_data_params)
            # #load data from api
            # failed_models = self.data_loader.run(self.model_data_params)
            self.list_model_data = self.finetune_model.generate_model_data()
            #finetune model
            model,dataset = self.finetune_model.run_finetune(self.list_model_data,self.config)

           
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

    main.load_datas()
    main.runtrain()
    
    manager = ConversationManager(
        model_name="beatajackowska_DialoGPT-RickBot",
        max_length=100,
        temperature=0.7,
        top_p=0.9
    )
    
    print(f"{Fore.CYAN}Starting conversation. Type 'quit' to exit.{Style.RESET_ALL}")
    
    while True:
        # Get user input
        user_input = input(f"{Fore.YELLOW}You: {Style.RESET_ALL}")
        if user_input.lower() == 'quit':
            break
        
        # Generate response
        response = manager.chat(user_input)
        if response:
            print(f"{Fore.GREEN}Assistant: {response}{Style.RESET_ALL}")
    
    # Save conversation history
    conversation_history = manager.get_memory()
    if conversation_history:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = main.HomePath / "conversation_history" / f"chat_{timestamp}.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_file, 'w') as f:
            json.dump(conversation_history, f, indent=2)
        print(f"{Fore.CYAN}Conversation history saved to {history_file}{Style.RESET_ALL}")

# What To do

#this just addon.
# create api for model
#app for multimodal

#main stuff
# create multimodal template
#merge dataset to multimodal
#create event loop for use input and model input (should recieve multiple input type data as sametime)
#create attention between event loop for filter unwanting data so runtime not interfere
#create embbeding and en-router-attention with de-router-attention (shared attention or embed)
#create function feed input from router to encoder_model or decoder model
#create function to display output by model output from router_attention
#####note output from model should be stream into input of model_input instead of user_input or its model input for inteferencing
#####the data should be on eventloop instead of model loop so crack that 1 bit llms