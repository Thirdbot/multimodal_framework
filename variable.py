from huggingface_hub import HfApi 
from pathlib import Path

class Variable():
    
    def __init__(self):
        self.WORKSPACE = Path(__file__).parent.absolute()      
        self.hf_api = HfApi()
        self.model_data_logs = dict()
        
        self.Api_card_file = 'ApiCardSet.json'
        
        self.configs_folder = 'configs'
        
        self.repositories_folder = 'repositories'
        self.repositories_dataset_folder = 'datasets'
        
        self.saved_configs_file = 'saved_config.json'
        
        self.DMConfig_DIR = self.WORKSPACE / self.configs_folder
        
        self.Card_Path =  self.DMConfig_DIR / self.Api_card_file
        
        self.REPO_DIR = self.WORKSPACE / self.repositories_folder
        
        self.DATASETS_DIR = self.REPO_DIR / self.repositories_dataset_folder
        
        self.SAVED_CONFIG_Path= self.DMConfig_DIR / self.saved_configs_file

