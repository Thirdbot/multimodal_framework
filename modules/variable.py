from huggingface_hub import HfApi 
from pathlib import Path
import torch

class Variable():
    
    def __init__(self):
        self.WORKSPACE = Path(__file__).parent.parent.absolute()      
        self.hf_api = HfApi()
        self.model_data_logs = dict()
        
        self.Api_card_file = 'ApiCardSet.json'
        
        self.configs_folder = 'configs'
        
        self.repositories_folder = 'repositories'
        self.repositories_dataset_folder = 'datasets'
        self.repositories_model_folder = 'models'
        self.custom_model_folder = "custom_models"
        self.vision_model_folder = "vision-model"
        self.conversation_model_folder = "conversation-model"
        self.model_saved_folder = "model-trained"
        self.model_checkpoints = "checkpoints"

        
        
        self.saved_configs_file = 'saved_config.json'
        
        self.DMConfig_DIR = self.WORKSPACE / self.configs_folder
        
        self.Card_Path =  self.DMConfig_DIR / self.Api_card_file
        
        self.REPO_DIR = self.WORKSPACE / self.repositories_folder
        
        self.DATASETS_DIR = self.REPO_DIR / self.repositories_dataset_folder
        self.LocalModel_DIR = self.REPO_DIR / self.repositories_model_folder
        
        self.SAVED_CONFIG_Path= self.DMConfig_DIR / self.saved_configs_file

         
        self.CUTOM_MODEL_DIR = self.WORKSPACE / self.custom_model_folder
        self.VISION_MODEL_DIR = self.CUTOM_MODEL_DIR / self.vision_model_folder
        self.REGULAR_MODEL_DIR = self.CUTOM_MODEL_DIR / self.conversation_model_folder        
        
        self.MODEL_DIR = self.WORKSPACE / self.model_saved_folder
        self.CHECKPOINT_DIR = self.WORKSPACE / self.model_checkpoints
        self.OFFLOAD_DIR = self.WORKSPACE / "offload"
        self.training_config_path = self.CUTOM_MODEL_DIR / "training_config.json"
        self.DATASET_FORMATTED_DIR = self.CUTOM_MODEL_DIR / "formatted_datasets"
        self.DTYPE = torch.float32

        self.chat_template_path = self.WORKSPACE / 'chat_template'
