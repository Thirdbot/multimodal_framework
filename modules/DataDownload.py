import json
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import hf_hub_download, HfApi
from pathlib import Path
from colorama import Fore, Style, init
import os
init(autoreset=True)

import shutil

from variable import Variable

''' Download locally as seperated folder for training and inference'''
class FlexibleDatasetLoader:
    def __init__(self, split='train'):
        self.variable = Variable()
        self.config = {}
        self.split = split
        self.dataset = None
        self.api = self.variable.hf_api
        
        # Set up local repository paths
        self.WORKSPACE_DIR = Path(__file__).parent.parent.absolute()
        self.REPO_DIR = self.variable.REPO_DIR
        self.DATASETS_DIR = self.variable.DATASETS_DIR
        self.DATAMODEL_DIR = self.variable.DMConfig_DIR
        self.SAVED_CONFIG_FILE = self.variable.SAVED_CONFIG_Path
    
    
    #load dataset
    def load(self, name, config):
        #load datasets with congig
        print('name:',name)
        print('config:',config)
        if config:
            try:
                # Create dataset directory with name and config
                short_name = name.split('/')[-1]  # Get just the last part of the name
                dataset_dir = self.DATASETS_DIR / f"{short_name}"
                dataset_dir.mkdir(parents=True, exist_ok=True)
                
                
                
                self.dataset = load_dataset(
                    name,
                    config,
                    split=self.split,
                    # cache_dir=dataset_dir
                )
                print(f"{Fore.GREEN}Successfully loaded dataset {name}{Style.RESET_ALL}")
               
               
               ### i forgot what it does so dont delete this.
                
            except Exception as e:
                print(f"{e}")
                #config error or entering wrong config then goes recursive to else statement
                return self.load(name, None)
                    
        else:
            #if there is no config or config None then find config
            configs = get_dataset_config_names(name)
            #always return configs
            if isinstance(configs, list):
                #instance of saved_config
                user_input = None
                print(f"{Fore.CYAN}Available configs for {name}: {configs}{Style.RESET_ALL}")
                saved_config = dict()
                try:
                    with open(self.SAVED_CONFIG_FILE, 'r') as f:
                        #load dict format of json format
                        saved_config = json.load(f)
                except:
                    #create new files
                    self.SAVED_CONFIG_FILE.touch(exist_ok=True)
                
                #find name table 
                if name in saved_config:
                    user_input = saved_config[name]
                else:
                    user_input = input(f"{Fore.YELLOW}Enter the config you want to use: {Style.RESET_ALL}")
                    #new config
                    saved_config[name] = user_input
                    self.config = saved_config
                    with open(self.SAVED_CONFIG_FILE, 'w') as f:
                        json.dump(saved_config, f, indent=4)
                #return config as it will be recursive again last times
                return self.load(name, user_input)
            
        # Get the dataset info to find the actual files
        dataset_info = self.api.dataset_info(name)
        # Download each file from the dataset
        for file_info in dataset_info.siblings:
            try:
                # Create the target directory if it doesn't exist
                target_dir = dataset_dir / os.path.dirname(file_info.rfilename)
                # target_dir.mkdir(parents=True, exist_ok=False)
                
                # # Use a shorter cache path
                # cache_dir = self.REPO_DIR / "cache" / short_name
                # cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Download directly to dataset directory
                file_path = hf_hub_download(
                    repo_id=name,
                    filename=file_info.rfilename,
                    repo_type="dataset",
                    local_dir=str(target_dir),
                    cache_dir=str(target_dir),
                    # force_download=True
                )
                # Copy the file to the target directory if it's not already there
                target_path = target_dir / os.path.basename(file_info.rfilename)
                if not target_path.exists():
                    shutil.copy2(file_path, target_path)
                print(f"{Fore.GREEN}Downloaded {file_info.rfilename} to: {target_path}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.YELLOW}Warning: Could not download {file_info.rfilename}: {str(e)}{Style.RESET_ALL}")
                continue
            
    def get(self):
        return self.dataset



class ModelLoader:
    def __init__(self):
        super().__init__()
        self.api = HfApi()
        self.variable = Variable()
        
        # Set up local repository paths for both model and datasets
        self.REPO_DIR = self.variable.REPO_DIR
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
                        print(f"{Fore.GREEN}Downloaded {file_info.rfilename} to: {file_path}{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.YELLOW}Warning: Could not download {file_info.rfilename}: {str(e)}{Style.RESET_ALL}")
                        continue
                    
        except Exception as e:
            print(f"{Fore.RED}Error downloading model {name}: {str(e)}{Style.RESET_ALL}")
            

class DataLoader():
    def __init__(self):

        self.model = ModelLoader()
        self.dataset = FlexibleDatasetLoader()
        
        # self.datadict = {'model':'',"datasets":[]}
        # self.saved_config = self.dataset.saved_config
        self.config = self.dataset.config
        

    def run(self,params):
      
        return self.load(params)
        
    def load(self,to_install):
        
        for model,datasets in to_install['model'].items():
            print('model:',model)
            print('dataset:',datasets)
            
            try:
                #load model to repository
                self.model.load_model(model)
                
                print(f"{Fore.CYAN}Processing model: {model} with datasets: {datasets}{Style.RESET_ALL}")

                if isinstance(datasets,dict):
                    try:
                        for dataset,info in datasets.items():
                                #load datasets to repository
                                self.dataset.load(dataset,self.config)
                    except Exception as e:
                        print(f"{Fore.RED}Error processing dataset {dataset}: {str(e)}{Style.RESET_ALL}")
                    
                
            except Exception as e:
                print(f"{Fore.RED}Error processing model {model}: {str(e)}{Style.RESET_ALL}")
    
   
            
    