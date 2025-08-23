#for create multimodal dataset from combination of datasets

import os
import json
from pathlib import Path
from modules.ApiDump import Manager as DatasetHandler
from modules.DataDownload import DataLoader
from pandas import pd

class CreateMultimodalDataset:
    def __init__(self):
        self.HomePath = Path(__file__).parent.absolute()
        
        self.DataModelFolder = f"{self.HomePath}/DataModel_config"
        Path(self.DataModelFolder).mkdir(parents=True, exist_ok=True)
        self.temporal_file_path = f'{self.DataModelFolder}/multimodal_data_model.json'
        Path(self.temporal_file_path).touch(exist_ok=True)
        
        self.datafile = 'installed.json'
        self.model_data_json_path = f"{self.DataModelFolder}/{self.datafile}"
        Path(self.model_data_json_path).touch(exist_ok=True)
        
        self.dataset_handler = DatasetHandler()
        self.data_loader = DataLoader()
        
        self.model_data_params = {
            "model_name": ["beatajackowska/DialoGPT-RickBot"],
            "datasets_name": ["OpenAssistant/oasst2","laion/laion400m"],
            "model_amount": 10,
            "datasets_amount": 10
        }
    
    def load_datasets(self):
        # Handle data configuration
        self.dataset_handler.handle_data(self.temporal_file_path, **self.model_data_params)
        # Load data from API
        failed_models = self.data_loader.run(self.model_data_params)
        return failed_models
    
    def format_data(self):
        #load data from multimodal_data_model.json
        installed = None
        if self.temporal_file_path.exists():
            if self.temporal_file_path.stat().st_size > 0:
                installed = pd.read_json(self.installed_filepath)


if __name__ == "__main__":
    dataset_creator = CreateMultimodalDataset()
    failed_models = dataset_creator.load_datasets()

