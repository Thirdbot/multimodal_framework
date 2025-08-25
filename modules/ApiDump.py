# from datasets import load_dataset
# import os
from huggingface_hub import HfApi,ModelInfo,DatasetInfo
from transformers import AutoModel
from datasets import load_dataset,get_dataset_split_names
import os
# import requests
# import json
from pathlib import Path
from typing import Optional,Iterable
# from pydantic import BaseModel, Field
# from datasets import load_dataset, get_dataset_config_names
# from colorama import Fore, Style, init
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry
# import time
# from huggingface_hub import HfApi, HfFolder
# from huggingface_hub.utils import HfHubHTTPError
import json
# # Initialize colorama
# init(autoreset=True)

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class ApiCardSetup:
    '''Setting Api calling Card as models and datasets be required'''
    def __init__(self):
        self.hf_api = HfApi()
        self.model_data_logs = dict()
        self.path = Path(__file__).parent.parent.absolute() / 'ApiCardSet.json'

    def set(self,list_models:Optional[Iterable[ModelInfo]],list_datasets:Optional[Iterable[DatasetInfo]]):
        '''request hf api and convert to history json by getting list_models object and list_datasets object.
        
            No Return (only update file)
        '''
       
       #empty List of model_name and datasets_name
        model_name_list:list = list()
        dataset_name_list:list = list()
        
        #format Path -> String
        str_literal_path = self.path.as_posix()
        
        #DataCard Form
        previous = {'model':dict(dict())}
        
        #Get Model Name in Card
        if list_models is not None:
            for model in list_models:
                model_name_list.append(model.id)
        #Get Dataset Name in Card
        if list_datasets is not None:
            for dataset in list_datasets:
                dataset_name_list.append(dataset.id)
        #Load Existed Card
        if os.path.exists(str_literal_path):
            with open(str_literal_path,'r') as f:
                previous = json.load(f)
        
        #Put Info in Card as it getting Non Duplicate in Model and Dataset Using Dict Type
        for model in model_name_list:
            
            previous['model'][model] = dict()
            model_layer = previous
            for data in dataset_name_list:
                model_layer['model'][model][data] = ""
        with open (str_literal_path,'w') as f:

            json.dump(previous,f, ensure_ascii=False, indent=4)
        
        #Return Newest Card
        return previous

                    
                
        

