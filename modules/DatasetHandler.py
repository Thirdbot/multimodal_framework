from datasets import load_dataset
import os
import requests
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from datasets import load_dataset, get_dataset_config_names
from colorama import Fore, Style, init
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from huggingface_hub import HfApi, HfFolder
from huggingface_hub.utils import HfHubHTTPError

# Initialize colorama
init(autoreset=True)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # increased number of retries
        backoff_factor=2,  # increased backoff time
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# -- get data from huggingface api
# need to add more function to get data from other api
# need to get category to save in json file each name/dataset
# need to refactor code to make it more efficient

class APIFetch(BaseModel):
    web_address: Optional[str] = Field(default=None)
    type: Optional[list] = Field(default=None)
    search: Optional[str] = Field(default=None)
    author: Optional[str] = Field(default=None)
    task_categories: Optional[list] = Field(default=[])
    tag: Optional[str] = Field(default=None)
    list_key:Optional[list] = Field(default=[])
    modality: Optional[str] = Field(default=None)
    model_name:list = Field(default=[])
    datasets_name:list = Field(default=[])

    
    
    def __init__(self, **data):
        super().__init__(**data)
        forbid_key = ['model_name','type','datasets_name','web_address','list_key']
        # Only set attributes that were actually passed in
        for key, value in data.items():
            if (value is not None and len(value) > 0) and key not in forbid_key:
                self.list_key.append(key)
                setattr(self,key,value)
        
                
                
   

    def concatenate_request(self):
        target_url = []
        data_list = []
        model_data_url = {'model':str,'datasets':[]}
        if not self.web_address:
            return None
        
        params = {}
        for type in self.type:
            web_address = self.web_address.rstrip(',')
            web_address = web_address + type

            if type == 'models':
                if len(self.model_name) >= 1:
                    for model_name in self.model_name:
                        web_address = web_address + '/' + model_name
                        model_data_url['model'] = web_address
                        target_url.append(model_data_url)
                        model_data_url = {'model':str,'datasets':[]}
                        web_address = self.web_address + type
                        
            elif type == 'datasets':   
                if len(self.model_name) > 1 and any(isinstance(i,list) for i in self.datasets_name):
                    for idx,datasets_name in enumerate(self.datasets_name):
                        for datasets_name in datasets_name:
                            web_address = self.web_address.rstrip(',') + type + '/' + datasets_name
                            data_list.append(web_address)
                        target_url[idx]['datasets'] = data_list
                        data_list = []
                    if len(target_url) > 1:
                        return target_url
                    
                elif len(self.model_name) > 1 and len(self.datasets_name) > 1:
                    for datasets_name in self.datasets_name:
                        web_address = self.web_address.rstrip(',') + type + '/' + datasets_name
                        data_list.append(web_address)
                    target_url = [{**trt, 'datasets': data_list} for trt in target_url]
                    data_list = []
                    if len(target_url) > 1:
                        return target_url
                
                elif len(self.model_name) > 1 and len(self.datasets_name) == 1:
                    for idx,model_name in enumerate(self.model_name):
                        web_address = self.web_address.rstrip(',') + type + '/' + self.datasets_name[0]
                        data_list.append(web_address)
                        target_url[idx]['datasets'] = data_list
                        data_list = []
                    if len(target_url) > 1:
                        return target_url

                elif len(self.model_name) == 1 and len(self.datasets_name) > 1:
                    for datasets_name in self.datasets_name:
                        web_address = self.web_address.rstrip(',') + type + '/' + datasets_name
                        data_list.append(web_address)
                    target_url[0]['datasets'] = data_list
                    return target_url
                    
                elif len(self.model_name) == 1 and len(self.datasets_name) == 1:
                    for datasets_name in self.datasets_name:
                        web_address = self.web_address.rstrip(',') + type + '/' + datasets_name
                        data_list.append(web_address)
                    target_url = [{**trt, 'datasets': data_list} for trt in target_url]
                    return target_url

                
                
            for key in self.list_key:
                params[key] = getattr(self,key)
            
            if params:
                print(f"{Fore.WHITE}Params: {params}{Style.RESET_ALL}")
                param_string = '&'.join(f"{k}={v}" for k, v in params.items() if k != "task_categories")
                    
                if len(self.task_categories) >= 1:
                    param_string = '&'.join(f"{k}={v}" for k, v in params.items() if k != "task_categories")
                    for idx ,task in enumerate(self.task_categories):
                        if type == 'models':
                            data = f"{web_address}?{param_string}&task_categories={task}"
                            print(f"{Fore.WHITE}Generated URL: {data}{Style.RESET_ALL}")
                            model_data_url['model'] = data
                            target_url.append(model_data_url)
                            model_data_url = {'model':str,'datasets':[]}
                        elif type == 'datasets':
                            data = f"{web_address}?{param_string}&task_categories={task}"
                            data_list.append(data)
                            target_url[idx]['datasets'] = data_list
                            data_list = []
                elif type == 'models':
                    model_data_url['model'] = web_address + '?' + param_string
                    target_url.append(model_data_url)
                    model_data_url = {'model':str,'datasets':[]}
                elif type == 'datasets':
                    data_list.append(web_address + '?' + param_string)
                    target_url[-1]['datasets'] = data_list
                    data_list = []
                
        return target_url

    def get_api_json(self):
        concatenated_json = []
        data_list = []
        url = self.concatenate_request()
        print(f"{Fore.YELLOW}Generated URLs: {url}{Style.RESET_ALL}")
        
        if not url:
            return None
        if isinstance(url,list):
            session = create_session()
            for idx,url in enumerate(url):
                if not isinstance(url,dict):
                    try:
                        response = session.get(url, timeout=30)  # 30 second timeout
                        print(f"{Fore.CYAN}Fetching data from: {url}{Style.RESET_ALL}")
                        response.raise_for_status()
                        concatenated_json.append(response.json())
                    except requests.exceptions.RequestException as e:
                        print(f"{Fore.RED}Error fetching data: {e}{Style.RESET_ALL}")
                        continue
                else:
                    try:
                        Home_dir = Path(__file__).parent.parent.absolute()
                        split_index = url['model'].split('/')
                        local_files_dir = split_index[-3:]
                        
                        if "custom_models" in local_files_dir:
                            Home_dir = str(Home_dir)
                            # Store relative path instead of absolute path
                            for path in local_files_dir:
                                Home_dir = os.path.join(Home_dir,path)
                            model_entry = {"model": Home_dir, "datasets": []}
                            concatenated_json.append(model_entry)
                            
                        else:
                            response = session.get(url['model'], timeout=30)  # 30 second timeout
                            print(f"{Fore.CYAN}Fetching model data from: {url['model']}{Style.RESET_ALL}")
                            response.raise_for_status()
                            model_response = response.json()
                            model_entry = {"model": model_response, "datasets": []}
                            concatenated_json.append(model_entry)
                        
                        for dataset_url in url['datasets']:
                            try:
                                response = session.get(dataset_url, timeout=30)  # 30 second timeout
                                print(f"{Fore.CYAN}Fetching dataset from: {dataset_url}{Style.RESET_ALL}")
                                response.raise_for_status()
                                dataset_response = response.json()
                                data_list.append(dataset_response)
                            except requests.exceptions.RequestException as e:
                                print(f"{Fore.RED}Error fetching dataset: {e}{Style.RESET_ALL}")
                                continue
                        
                        concatenated_json[idx]['datasets'] = data_list
                        data_list = []
                        
                    except requests.exceptions.RequestException as e:
                        print(f"{Fore.RED}Error fetching model data: {e}{Style.RESET_ALL}")
                        continue
            return concatenated_json
        else:
            try:
                session = create_session()
                response = session.get(url, timeout=30)  # 30 second timeout
                print(f"{Fore.CYAN}Fetching data from: {url}{Style.RESET_ALL}")
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"{Fore.RED}Error fetching data: {e}{Style.RESET_ALL}")
                return None


class Convert(BaseModel):
    data_model:Optional[list] = Field(default=None)
    keyword:Optional[str] = Field(default='id')
    list_key:Optional[list] = Field(default=[])
    model_amount:Optional[int] = Field(default=1)
    datasets_amount:Optional[int] = Field(default=1)
    


    def __init__(self, **data):
        super().__init__(**data)
        forbid_key = ['keyword','model_amount','datasets_amount']
        for key,value in data.items():
            if value is not None and key not in forbid_key:
                self.list_key.append(key)
                setattr(self,key,value)
        
    def convert_to_json(self,file_path):
        name_list = []
        i=0
        
        for key in self.list_key:
            for value in getattr(self, key):
                if key == 'data_model':
                    # Create a new model entry
                    if isinstance(value['model'],list):
                        dataset_list = value['datasets']
                        for idx,model in enumerate(value['model']):
                            if isinstance(model,dict):
                                model_entry = {
                                    'model':model.get(self.keyword,''),
                                    'datasets': []
                                }
                            else:
                                model_entry = {
                                    'model':str(model),
                                    'datasets': []
                                }
                            i += 1
                            
                            # Handle datasets
                            for dataset in dataset_list:
                                if isinstance(dataset, dict):
                                    dataset_id = dataset.get(self.keyword, '')
                                else:
                                    dataset_id = str(dataset)
                                
                                dataset_used = False
                                for prev_entry in name_list[:idx]:
                                    if dataset_id in prev_entry['datasets']:
                                        dataset_used = True
                                        break
                                
                                if not dataset_used and (self.datasets_amount is None or len(model_entry['datasets']) < self.datasets_amount):
                                    model_entry['datasets'].append(dataset_id)
                            
                            if self.model_amount is None or i < self.model_amount:
                                name_list.append(model_entry)
                    else:
                        if isinstance(value['model'],dict):
                            model_entry = {
                                'model':value['model'].get(self.keyword,''),
                                'datasets': []
                            }
                        else:
                            model_entry = {
                                'model':str(value['model']),
                                'datasets': []
                            }
                    
                        # Add all datasets for this model
                        for dataset in value['datasets']:
                            if isinstance(dataset, dict):
                                dataset_id = dataset.get(self.keyword, '')
                            else:
                                dataset_id = str(dataset)
                            
                            if self.datasets_amount is None or len(model_entry['datasets']) < self.datasets_amount:
                                model_entry['datasets'].append(dataset_id)
                    
                        # Add the model entry if we haven't reached the limit
                        if len(name_list) < self.model_amount:
                            name_list.append(model_entry)
        
        # Write the final result to file
        with open(file_path, 'w+') as f:
            json.dump(name_list, f, indent=4)
        return name_list
 
        

class Manager:
    def __init__(self):
        self.data_type = ["models","datasets"]
        self.session = create_session()
        # Initialize Hugging Face API
        self.hf_api = HfApi()
        # Set longer timeout for dataset operations
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 minutes timeout
    
    def handle_data(self,file_path,model_name:Optional[list]=[],
                    datasets_name:Optional[list]=[],task:Optional[list]=[],
                    model_amount:Optional[int]=1,datasets_amount:Optional[int]=1,
                    modality:Optional[str]=None,search:Optional[str]=None):
        
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{Fore.RED}File {file_path} has been deleted.{Style.RESET_ALL}")
            Path(file_path).touch(exist_ok=True)
            print(f"{Fore.GREEN}File {file_path} has been created.{Style.RESET_ALL}")
        else:
            Path(file_path).touch(exist_ok=True)
            print(f"{Fore.GREEN}File {file_path} has been created.{Style.RESET_ALL}")
        # task = "text-generation"
        
        model_api = APIFetch(
            web_address="https://huggingface.co/api/",
            type=self.data_type,
            task_categories=task,
            model_name=model_name,
            datasets_name=datasets_name,
            modality="modality%3A"+modality if modality else None,
            search=search,
            # task_categories=['text-generation','image-generation'],
            # author="huggingface",
            # model_name=["beatajackowska/DialoGPT-RickBot"],
            # datasets_name=["theneuralmaze/rick-and-morty-transcripts-sharegpt"]
        )
        
        all_model_name = model_api.get_api_json()
        converter = Convert(data_model=all_model_name,
                          keyword="id",
                          model_amount=model_amount,
                          datasets_amount=datasets_amount)
        return converter.convert_to_json(file_path)
        
        

