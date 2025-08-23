from datasets import load_dataset
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
from colorama import Fore, Style, init
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry
# import time
# from huggingface_hub import HfApi, HfFolder
# from huggingface_hub.utils import HfHubHTTPError
import json
# # Initialize colorama
init(autoreset=True)

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class ApiCardSetup:
    def __init__(self):
        self.hf_api = HfApi()
        self.model_data_logs = dict()
        self.path = Path(__file__).parent.parent.absolute() / 'ApiCardSet.json'

    def set(self,list_models:Optional[Iterable[ModelInfo]],list_datasets:Optional[Iterable[DatasetInfo]]):
        '''request hf api and convert to history json by getting list_models object and list_datasets object.
        
            No Return (only update file)
        '''
        
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        #     print(f"{Fore.RED}File {file_path} has been deleted.{Style.RESET_ALL}")
        #     Path(file_path).touch(exist_ok=True)
        #     print(f"{Fore.GREEN}File {file_path} has been created.{Style.RESET_ALL}")
        # else:
        #     Path(file_path).touch(exist_ok=True)
        #     print(f"{Fore.GREEN}File {file_path} has been created.{Style.RESET_ALL}")
        # task = "text-generation"

        # acess_token = os.environ.get("hf_token")

        # list_models = self.hf_api.list_models(tags="text-generation",limit=1,gated=False,language='thai')
        # list_datasets = self.hf_api.list_datasets(dataset_name='tlcv2.0_oa',limit=1,gated=False)

        model_name_list:list = list()
        dataset_name_list:list = list()

        str_literal_path = self.path.as_posix()
        
        previous = {'model':dict(dict())}
        
        if list_models is not None:
            for model in list_models:
                model_name_list.append(model.id)
        if list_datasets is not None:
            for dataset in list_datasets:
                dataset_name_list.append(dataset.id)
        
        if os.path.exists(str_literal_path):
            with open(str_literal_path,'r') as f:
                previous = json.load(f)
        else:
            pass
            # self.path.touch(exist_ok=True)
        # previous = json.load(previous)  # type: ignore
        print(previous)
        for model in model_name_list:
            
            previous['model'][model]
            model_layer = previous
            for data in dataset_name_list:
                model_layer['model'][model][data] = ""
        print(previous)
        with open (str_literal_path,'w') as f:
            # y = json.dumps(dict(previous))
            # print(y)
            json.dump(previous,f, ensure_ascii=False, indent=4)
        

                    
                
        

