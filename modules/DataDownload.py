#use this to download the data from the data_model.json and successully save to installed.json and sent model to finetuned for use again in main.py
import pandas as pd 
import json
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import hf_hub_download, HfApi
from pathlib import Path
from colorama import Fore, Style, init
from modules.ApiDump import ApiCardSetup
from transformers import AutoModelForCausalLM
import os
import requests
from urllib.parse import urlparse

# Initialize colorama
init(autoreset=True)

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



''' Download locally as seperated folder for training and inference'''
class FlexibleDatasetLoader:
    def __init__(self, split='train'):

        self.config = None
        self.split = split
        self.saved_config = {}  # Changed to dict to store configs per dataset
        self.dataset = None
        self.file_paths = {}
        self.api = HfApi()
        
        # Set up local repository paths
        self.WORKSPACE_DIR = Path(__file__).parent.parent.absolute()
        self.REPO_DIR = self.WORKSPACE_DIR / "repositories"
        self.DATASETS_DIR = self.REPO_DIR / "datasets"
        self.DATAMODEL_DIR = self.WORKSPACE_DIR / "DataModel_config"
        self.SAVED_CONFIG_FILE = self.DATAMODEL_DIR / "saved_config.json"
        
    def load(self, name, config):
        if config is not None:
            try:
                # Create dataset directory with name and config
                short_name = name.split('/')[-1]  # Get just the last part of the name
                dataset_dir = self.DATASETS_DIR / f"{short_name}"
                dataset_dir.mkdir(parents=True, exist_ok=False)
                
                # Get the dataset info to find the actual files
                dataset_info = self.api.dataset_info(name)
                
                self.dataset = load_dataset(
                    name,
                    config,
                    split=self.split,
                    # cache_dir=dataset_dir
                )
                print(f"{Fore.GREEN}Successfully loaded dataset {name}{Style.RESET_ALL}")
                self.config = config
                self.saved_config[name] = config
               
               
               ### i forgot what it does so dont delete this.
                    
                # Download each file from the dataset
                for file_info in dataset_info.siblings:
                    try:
                        # Create the target directory if it doesn't exist
                        target_dir = dataset_dir / os.path.dirname(file_info.rfilename)
                        target_dir.mkdir(parents=True, exist_ok=False)
                        
                        # # Use a shorter cache path
                        # cache_dir = self.REPO_DIR / "cache" / short_name
                        # cache_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Download directly to dataset directory
                        file_path = hf_hub_download(
                            repo_id=name,
                            filename=file_info.rfilename,
                            repo_type="dataset",
                            local_dir=str(target_dir),
                            # cache_dir=str(cache_dir),
                            # force_download=True
                        )
                        # Copy the file to the target directory if it's not already there
                        target_path = target_dir / os.path.basename(file_info.rfilename)
                        if not target_path.exists():
                            import shutil
                            shutil.copy2(file_path, target_path)
                        self.file_paths[file_info.rfilename] = str(target_path)
                        print(f"{Fore.GREEN}Downloaded {file_info.rfilename} to: {target_path}{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.YELLOW}Warning: Could not download {file_info.rfilename}: {str(e)}{Style.RESET_ALL}")
                        continue
                
            except Exception as e:
                if name in self.saved_config:
                    return self.load(name, self.saved_config[name])
                return self.load(name, None)
                    
        else:
            configs = get_dataset_config_names(name)
            if isinstance(configs, list):
                saved_config = {}
                user_input = None
                print(f"{Fore.CYAN}Available configs for {name}: {configs}{Style.RESET_ALL}")
                try:
                    with open(self.SAVED_CONFIG_FILE, 'r') as f:
                        saved_config = json.load(f)
                except:
                    self.SAVED_CONFIG_FILE.touch(exist_ok=True)
                if name in saved_config:
                    user_input = saved_config[name]
                else:
                    user_input = input(f"{Fore.YELLOW}Enter the config you want to use: {Style.RESET_ALL}")
                    self.config = user_input
                    self.saved_config[name] = user_input
                    saved_config.update(self.saved_config)
                    with open(self.SAVED_CONFIG_FILE, 'w') as f:
                        json.dump(saved_config, f, indent=4)
                return self.load(name, user_input)

    def get(self):
        return self.dataset

    def get_local_files(self):
        """Get the paths to the locally downloaded files"""
        return self.file_paths

class ModelLoader:
    def __init__(self):
        super().__init__()
        self.file_paths = {}
        self.api = HfApi()
        
        # Set up local repository paths for both model and datasets
        self.WORKSPACE_DIR = Path(__file__).parent.parent.absolute()
        self.REPO_DIR = self.WORKSPACE_DIR / "repositories"
        self.REPO_DIR.mkdir(parents=True, exist_ok=True)
        
    def load_model(self, name):
        try:
            # Get the model info to find the actual files
            local_path = Path(name)
            if local_path.exists() :
                # For custom models, use the relative path from the workspace
                print(f"{Fore.GREEN}Using custom model from: {local_path}{Style.RESET_ALL}")
            else:
                model_info = self.api.model_info(name)
                
                # Download each file from the model
                for file_info in model_info.siblings:
                    try:
                        file_path = hf_hub_download(
                            repo_id=name,
                            filename=file_info.rfilename,
                            repo_type="model",
                            cache_dir=self.REPO_DIR
                        )
                        self.file_paths[file_info.rfilename] = file_path
                        print(f"{Fore.GREEN}Downloaded {file_info.rfilename} to: {file_path}{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.YELLOW}Warning: Could not download {file_info.rfilename}: {str(e)}{Style.RESET_ALL}")
                        continue
                    
        except Exception as e:
            print(f"{Fore.RED}Error downloading model {name}: {str(e)}{Style.RESET_ALL}")
            
    def get_path(self):
        return self.file_paths

class DataLoader():
    def __init__(self):
        super().__init__()
        self.WORKSPACE_DIR = Path(__file__).parent.parent.absolute()
        self.DMConfig_DIR = self.WORKSPACE_DIR / 'DataModel_config'
        self.data_type = ["models","datasets"]

        self.DMConfig_DIR.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        self.datamodel_file = 'data_model.json'
        self.datamodel_filepath = self.DMConfig_DIR / self.datamodel_file
        self.installed_file = 'installed.json'
        self.installed_filepath = self.DMConfig_DIR / self.installed_file
        self.datamodel_filepath.touch(exist_ok=True)
        self.installed_filepath.touch(exist_ok=True)

        self.model = ModelLoader()
        self.dataset = FlexibleDatasetLoader()
        
        # self.datadict = {'model':'',"datasets":[]}
        self.saved_config = self.dataset.saved_config
        self.config = self.dataset.config
        
        # self.load(self.datamodel)
    
    def run(self,params):
        # datamodel = self.datamodel_load(params)
        return self.load(params)
        
    # def datainstall_load(self):
    #     if self.installed_filepath.exists():
    #         if self.installed_filepath.stat().st_size > 0:
    #             installed = pd.read_json(self.installed_filepath)
    #             return installed
    #         else:
    #             return pd.DataFrame(columns=['model','datasets'])

    # def datamodel_load(self,params):
    #     if self.datamodel_filepath.exists():
    #         if self.datamodel_filepath.stat().st_size > 0:
    #             df = pd.read_json(self.datamodel_filepath)
    #             return df
    #         else:
    #             print(f"{Fore.YELLOW}No data model found. Creating new one...{Style.RESET_ALL}")
    #             manager = ApiCardSetup()
    #             return manager.set(params)


    def load(self,to_install):
        # base_df = pd.DataFrame(install)
        # print(base_df['datasets'])
        
        for model,datasets in to_install['model'].items():
            print('model:'+model)
            print('dataset:',datasets)
            
            try:
                download_model = self.model.load_model(model)
                
                print(f"{Fore.CYAN}Processing model: {model} with datasets: {datasets}{Style.RESET_ALL}")

                if isinstance(datasets,dict):
                    try:
                        for dataset,info in datasets.items():
                                self.dataset.load(dataset,self.config)
                    except Exception as e:
                        print(f"{Fore.RED}Error processing dataset {dataset}: {str(e)}{Style.RESET_ALL}")
                # else:
                #     self.dataset.load(datasets,self.config)
                    
                

            except Exception as e:
                print(f"{Fore.RED}Error processing model {model}: {str(e)}{Style.RESET_ALL}")
    
   
            
    