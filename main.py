# # import os
# # import gc
# # import json
# # from datetime import datetime
# from pathlib import Path
# # from typing import Optional, Dict, Any

# # # Set environment variables before any other imports
# # os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# # os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
# # os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism

# # import torch
# # import yaml
# # from accelerate.utils import set_seed
# # from colorama import Fore, Style, init
# # from transformers import (
# #     AutoModelForCausalLM,
# #     AutoTokenizer,
# #     BitsAndBytesConfig,
# #     AutoConfig
# # )

# # from modules.defect import Report

# #download dataset and model from history logs
# from modules.DataDownload import DataLoader

# # api fetch handler to grab details as name then store in history logs
# from modules.DatasetHandler import Manager as DatasetHandler


# from modules.finetuning_model import Manager as FinetuneModel
# # from modules.interference import ConversationManager
# # from modules.chatTemplate import ChatTemplate
# # from modules.createbasemodel import CreateModel
# # import argparse

# # # Initialize colorama
# # init(autoreset=True)

# # # Set PyTorch settings
# # torch.set_num_threads(1)
# # torch.set_num_interop_threads(1)

# # # Define paths
# WORKSPACE_DIR = Path(__file__).parent.absolute()
# MODEL_DIR = WORKSPACE_DIR / "models" / "Text-Text-generation"
# OFFLOAD_DIR = WORKSPACE_DIR / "offload"
# OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

# # # Set random seed for reproducibility
# # set_seed(42)

# class Main:
#     def __init__(self):
#         """Initialize the main application class."""
#         self.HomePath = Path(__file__).parent.absolute()
#         self._setup_directories()
#         # self._initialize_components()
#         # self._setup_model_params()
        
#     def _setup_directories(self):
#         """Set up required directories and files."""
#         self.DataModelFolder = self.HomePath / "DataModel_config"
#         self.DataModelFolder.mkdir(parents=True, exist_ok=True)
        
#         self.temporal_file_path = self.DataModelFolder / "data_model.json"
#         self.temporal_file_path.touch(exist_ok=True)
        
#         self.model_data_json_path = self.DataModelFolder / "installed.json"
#         self.model_data_json_path.touch(exist_ok=True)
        
    
#     '''set up folders and class components'''
#     def _initialize_components(self):
#         """Initialize all required components."""
#         self.dataset_handler = DatasetHandler()
#         self.data_loader = DataLoader()
#         self.finetune_model = FinetuneModel(model_data_json_path=str(self.model_data_json_path))
#         self.chat_template = None
    
#     ''' set up parametes for runnning later. '''
#     def _setup_model_params(self):
#         """Set up model parameters."""
#         #need to fix this because api calling is a mess
#         self.model_data_params = {
#             "model_name": ["kyutai/helium-1-2b"],
#             # "model_name":["custom_models/text-generation/model-1"],
#             # "datasets_name": ["pythainlp/han-instruct-dataset-v4.0"],
#             "datasets_name":['pythainlp/han-instruct-dataset-v4.0'],
#             "model_amount": 2,
#             "datasets_amount": 2,
#         }
#         self.config = {}

#     '''fetch api from requirements'''
#     def load_datas(self):
#         """Load and process datasets."""
#         self.dataset_handler.handle_data(self.temporal_file_path, **self.model_data_params)
#         self.data_loader.run(self.model_data_params)
#         self.config = self.data_loader.saved_config
#         # print(f"{Fore.CYAN}Dataset Config:{Style.RESET_ALL} {self.config}")
    
# #     def runtrain(self):
# #         """Run the training process."""
# #         model = None
# #         dataset = None
# #         failed_models = None
        
# #         try:
# #             self.list_model_data = self.finetune_model.generate_model_data()
# #             model, dataset = self.finetune_model.run_finetune(self.list_model_data, self.config)
# #         except Exception as e:
# #             self._handle_training_error(e, model, dataset, failed_models)
    
# #     def _handle_training_error(self, error: Exception, model: Any, dataset: Any, failed_models: Any):
# #         """Handle training errors and generate reports."""
# #         report = Report()
# #         if model and dataset is not None:
# #             report.store_problem(model=model, dataset=dataset)
# #         else:
# #             print(Fore.RED + f"Error occurred: {error} but no report generated." + Style.RESET_ALL)
        
# #         if failed_models is not None:
# #             for element in failed_models:
# #                 report.store_problem(model=element['model'], dataset=element['datasets'])

# # def run_conversation():
# #     Home_dir = Path(__file__).parent.absolute()
# #     model_path = Home_dir / "models" / "text-generation" / "kyutai_helium-1-2b"
# #     """Run the conversation loop."""
# #     manager = ConversationManager(
# #         model_name=model_path,
# #         max_length=100,
# #         temperature=0.9,
# #         top_p=0.95
# #     )
    
# #     print(f"{Fore.CYAN}Starting conversation. Type 'quit' to exit.{Style.RESET_ALL}")
    
# #     while True:
# #         user_input = input(f"{Fore.YELLOW}You: {Style.RESET_ALL}")
# #         if user_input.lower() == 'quit':
# #             break
        
# #         response = manager.chat(user_input)
# #         if response:
# #             print(f"{Fore.GREEN}Assistant: {response}{Style.RESET_ALL}")
    
# #     save_conversation_history(manager)

# # def save_conversation_history(manager: ConversationManager):
# #     """Save the conversation history to a file."""
# #     conversation_history = manager.get_memory()
# #     if conversation_history:
# #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #         history_file = WORKSPACE_DIR / "conversation_history" / f"chat_{timestamp}.json"
# #         history_file.parent.mkdir(parents=True, exist_ok=True)
        
# #         with open(history_file, 'w') as f:
# #             json.dump(conversation_history, f, indent=2)
# #         print(f"{Fore.CYAN}Conversation history saved to {history_file}{Style.RESET_ALL}")

# # if __name__ == "__main__":
# #     main = Main()
   
    
# #     # create_model = CreateModel(model_name="model-1",model_category="text-generation")
    
# #     main.load_datas()
# #     main.runtrain()
# #     run_conversation()

# print('helloworld!')

# # Load the dataset type 1 Conversations column or instruction column
#     # dataset = load_dataset("theneuralmaze/rick-and-morty-transcripts-sharegpt", split="train")
    
#     #load the dataset type 2 regular text columns has an instruction data just like the dataset 1 but not naming its as conversations
#     # dataset = load_dataset("Menlo/high-quality-text-only-instruction", "default", split="train")
    
#     #load the dataset type 3 regular text columns seperated user and assistant with instruction data
#     # dataset = load_dataset("alexgshaw/natural-instructions-prompt-rewards", split="train")
#     #load the dataset type 4 regular text columns seperated user and assistant without instruction data
#     # dataset = load_dataset("AnonymousSub/MedQuAD_47441_Context_Question_Answer_Triples", split="train")
    
#     #instruction seperate column that has chosen and rejected data
#     # dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    

# #this file task

# #run on runpod

# #load model and dataset --prepare

# #train model with options llm,dllm,diffuser





# #[[global task]]

# # What To do

# # 1. redesign api request
# # 2. redesign model-dataset relation loop
# # 3. use vllm for model inference and still huggingface compatible for model push
# # 4. be able to update new repository as a docker container for run on big gpu
# # 5. each file independent and can be run with argument

# #make model seperate tokenizer and model
# #renew interference utilize langchain model
# # #try cut conner of chat template and create or utilize model's tokenizer to make dataste compatible with model include add model's compatible pipeline for multimodal


# #handle various of datasets downloadble files need each column
# #create template dataset
# #auto fallback dataset load request when failed to tokenize a dataset in case of naming convension

# #this just addon.
# # create api for model
# #app for multimodal

# #main stuff
# # create multimodal template
# #merge dataset to multimodal
# #create event loop for use input and model input (should recieve multiple input type data as sametime)
# #create attention between event loop for filter unwanting data so runtime not interfere
# #create embbeding and en-router-attention with de-router-attention (shared attention or embed)
# #create function feed input from router to encoder_model or decoder model
# #create function to display output by model output from router_attention
# #####note output from model should be stream into input of model_input instead of user_input or its model input for inteferencing
# #####the data should be on eventloop instead of model loop so crack that 1 bit llms


# import huggingface_hub
import os
from huggingface_hub import HfApi
# from datasets import load_dataset,get_dataset_split_names
from modules.ApiDump import ApiCardSetup
from modules.DataDownload import DataLoader
from modules.finetuning_model import FinetuneModel

acess_token = os.environ.get("hf_token")

api = HfApi()

setcard = ApiCardSetup()

list_models = api.list_models(tags="text-generation",limit=1,gated=False,language='thai')
# list_datasets = api.list_datasets(tags='text-generation',limit=3,gated=False)
list_datasets = api.list_datasets(dataset_name='FreedomIntelligence/medical-o1-reasoning-SFT',limit=1,gated=False)

#set new list to download
list_download = setcard.set(list_models,list_datasets)

downloading = DataLoader()

#download from datacard
downloading.run(list_download)


#finetune model
finetune = FinetuneModel()


