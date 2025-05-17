#use this to download the data from the data_model.json and successully save to installed.json and sent model to finetuned for use again in main.py
import pandas as pd 
import json
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import hf_hub_download, HfApi
from pathlib import Path
from colorama import Fore, Style, init
from modules.DatasetHandler import APIFetch, Convert
from modules.DatasetHandler import Manager
from transformers import AutoModelForCausalLM
import os
import requests
from urllib.parse import urlparse

# Initialize colorama
init(autoreset=True)

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class FlexibleDatasetLoader:
    def __init__(self, split='train', trust_remote_code=True):
        self.trust_remote_code = trust_remote_code
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
        
    def load(self, name, config):
        if config is not None:
            try:
                # Create dataset directory with name and config
                dataset_dir = self.DATASETS_DIR / f"{name.replace('/', '_')}_{config}"
                dataset_dir.mkdir(parents=True, exist_ok=True)
                
                # Get the dataset info to find the actual files
                dataset_info = self.api.dataset_info(name)
                
                self.dataset = load_dataset(
                    name,
                    config,
                    split=self.split,
                    trust_remote_code=self.trust_remote_code
                )
                print(f"{Fore.GREEN}Successfully loaded dataset {name} with config {config}{Style.RESET_ALL}")
                self.config = config
                self.saved_config[name] = config
               
                    
                # Download each file from the dataset
                for file_info in dataset_info.siblings:
                    try:
                        # Create the target directory if it doesn't exist
                        target_dir = dataset_dir / os.path.dirname(file_info.rfilename)
                        target_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Download directly to dataset directory
                        file_path = hf_hub_download(
                            repo_id=name,
                            filename=file_info.rfilename,
                            repo_type="dataset",
                            local_dir=str(dataset_dir),
                            local_dir_use_symlinks=False
                        )
                        self.file_paths[file_info.rfilename] = file_path
                        print(f"{Fore.GREEN}Downloaded {file_info.rfilename} to: {file_path}{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.YELLOW}Warning: Could not download {file_info.rfilename}: {str(e)}{Style.RESET_ALL}")
                        continue
                
            except Exception as e:
                if name in self.saved_config:
                    return self.load(name, self.saved_config[name])
                return self.load(name, None)
                    
        else:
            configs = get_dataset_config_names(name, trust_remote_code=self.trust_remote_code)
            if isinstance(configs, list):
                print(f"{Fore.CYAN}Available configs for {name}: {configs}{Style.RESET_ALL}")
                user_input = input(f"{Fore.YELLOW}Enter the config you want to use: {Style.RESET_ALL}")
                self.config = user_input
                self.saved_config[name] = user_input
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
        
        # Set up local repository paths
        self.WORKSPACE_DIR = Path(__file__).parent.parent.absolute()
        self.REPO_DIR = self.WORKSPACE_DIR / "repositories"
        self.REPO_DIR.mkdir(parents=True, exist_ok=True)
        
    def load_model(self, name):
        try:
    
            # Get the model info to find the actual files
            model_info = self.api.model_info(name)
            
            # Download each file from the model
            for file_info in model_info.siblings:
                try:
                    file_path = hf_hub_download(
                        repo_id=name,
                        filename=file_info.rfilename,
                        repo_type="model"
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
        
        self.datadict = {'model':'',"datasets":[]}
        self.saved_config = self.dataset.saved_config
        self.config = self.dataset.config
        
        # self.load(self.datamodel)
    
    def run(self,params):
        datamodel = self.datamodel_load(params)
        return self.load(datamodel)
        
    def datainstall_load(self):
        if self.installed_filepath.exists():
            if self.installed_filepath.stat().st_size > 0:
                installed = pd.read_json(self.installed_filepath)
                return installed
            else:
                return pd.DataFrame(columns=['model','datasets'])

    def datamodel_load(self,params):
        if self.datamodel_filepath.exists():
            if self.datamodel_filepath.stat().st_size > 0:
                df = pd.read_json(self.datamodel_filepath)
                return df
            else:
                print(f"{Fore.YELLOW}No data model found. Creating new one...{Style.RESET_ALL}")
                manager = Manager()
                return manager.handle_data(self.datamodel_filepath,params)


    def load(self,install, depth=0):
        installed = []
        failed_models = []  # Initialize failed_models at the start
        datadict = {'model':str,'datasets':[]}
        base_df = pd.DataFrame(install)
        compare_df = pd.DataFrame(self.datainstall_load())
        
        if depth > 3:
            print(f"{Fore.RED}Maximum retry depth reached.{Style.RESET_ALL}")
            return failed_models
        if base_df.equals(compare_df):
            print(f"{Fore.GREEN}Data already installed{Style.RESET_ALL}")
            for i, row in compare_df.iterrows():
                try:
                    model = row['model']
                    datasets = row['datasets']
                    print(f"{Fore.CYAN}Processing model: {model} with datasets: {datasets}{Style.RESET_ALL}")

                    if isinstance(datasets,list):
                        for dataset in datasets:
                            self.config = self.dataset.load(dataset,self.config)
                    else:
                        self.config = self.dataset.load(datasets,self.config)
                except Exception as e:
                    print(f"{Fore.RED}Error processing model as installed {model}: {str(e)}{Style.RESET_ALL}")
                    failed_models.append(row)
            return failed_models
        else:
            base_df['datasets'] = base_df['datasets'].apply(lambda x: sorted(x))
            compare_df['datasets'] = compare_df['datasets'].apply(lambda x: sorted(x))
            base_df['__row__'] = base_df.apply(lambda row: json.dumps(row.to_dict(), sort_keys=True), axis=1)
            compare_df['__row__'] = compare_df.apply(lambda row: json.dumps(row.to_dict(), sort_keys=True), axis=1)
            
            diff_model = base_df[~base_df['__row__'].isin(compare_df['__row__'])]
            diff_model = diff_model.drop(columns='__row__').to_dict(orient='records')
            diff_model = pd.DataFrame(diff_model)

            if diff_model.empty:
                print(f"{Fore.YELLOW}No new models to install{Style.RESET_ALL}")
                try:
                    for i, row in compare_df.iterrows():
                        model = row['model']
                        datasets = row['datasets']
                        print(f"{Fore.CYAN}Processing model: {model} with datasets: {datasets}{Style.RESET_ALL}")

                        if isinstance(datasets,list):
                            for dataset in datasets:
                                self.config = self.dataset.load(dataset,self.config)
                        else:
                            self.config = self.dataset.load(datasets,self.config)
                except Exception as e:
                    print(f"{Fore.RED}Error processing model {model}: {str(e)}{Style.RESET_ALL}")
                    failed_models.append(row)
            for i, row in diff_model.iterrows():
                try:
                    model = row['model']
                    download_model = self.model.load_model(model)
                    datasets = row['datasets']
                    print(f"{Fore.CYAN}Processing model: {model} with datasets: {datasets}{Style.RESET_ALL}")

                    if isinstance(datasets,list):
                        for dataset in datasets:
                             self.dataset.load(dataset,self.config)
                    else:
                        self.dataset.load(datasets,self.config)
                        
                    
                    datadict = {
                        'model': model,
                        'datasets': datasets
                  
                    }
                    installed.append(datadict)
                    print(installed)

                except Exception as e:
                    print(f"{Fore.RED}Error processing model {model}: {str(e)}{Style.RESET_ALL}")
                    failed_models.append(row)
                    
            if installed:
                prev_df = pd.DataFrame(self.datainstall_load())
                new_installs_df = pd.DataFrame(installed)
                result_df = pd.concat([prev_df, new_installs_df], ignore_index=True)
                with open(self.installed_filepath, 'w') as f:
                    json.dump(result_df.to_dict(orient='records'), f, indent=4)
                print(f"{Fore.GREEN}Successfully updated installed.json{Style.RESET_ALL}")

            if failed_models:
                print(f"{Fore.YELLOW}Retrying {len(failed_models)} failed installs...{Style.RESET_ALL}")
                self.load(failed_models, depth=depth + 1)
            
            return failed_models  # Always return failed_models

if __name__ == "__main__":
    from DatasetHandler import APIFetch,Convert
    loader = DataLoader()
    
   
            
    