from datasets import load_dataset
import os
import requests
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class APIFetch(BaseModel):
    web_address: Optional[str] = Field(default=None)
    type: Optional[str] = Field(default=None)
    search: Optional[str] = Field(default=None)
    author: Optional[str] = Field(default=None)
    task: Optional[str] = Field(default=None)
    tag: Optional[str] = Field(default=None)

    
    model_name:Optional[list] = Field(default=[])
    datasets_name:Optional[list] = Field(default=[])

    
    
    def __init__(self, **data):
        super().__init__(**data)
        # Only set attributes that were actually passed in
        for key, value in data.items():
            if value is not None:  # Only set non-None values
                setattr(self, key, value)
   

    def concatenate_request(self):
        forbid_key = ['model_name','datasets_name']
        list_model_name = []
        list_datasets_name = []
        if not self.web_address:
            return None
        
        web_address = self.web_address.rstrip(',')
        web_address = self.web_address + self.type
        
        for model_name in self.model_name:
            web_address = web_address + '/' + model_name
            list_model_name.append(web_address)
        for datasets_name in self.datasets_name:
            web_address = web_address + '/' + datasets_name
            list_datasets_name.append(web_address)
        
        if len(list_model_name) > 0:
            return list_model_name
        elif len(list_datasets_name) > 0:
            return list_datasets_name
        
        
        
        
        
        # Get all non-None attributes that were actually set
        params = {}
        for key , value in self.__dict__.items():
            if key != 'web_address' and key not in forbid_key and value is not None:  # Only include if it was actually set
                params[key] = value
        

        # Construct URL with parameters
        if params:
            param_string = '&'.join(f"{k}={v}" for k, v in params.items())
            return f"{web_address}?{param_string}"
        else:
            return web_address
       
    def get_api_json(self):
        concatenated_json = []
        url = self.concatenate_request()
        if not url:
            return None
        if isinstance(url,list):
            for url in url:
                try:
                    response = requests.get(url)
                    print(f"Generated URL: {url}")
                    response.raise_for_status()
                    concatenated_json.append(response.json())
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching data: {e}")
                    continue
            return concatenated_json
        else:
            try:
                response = requests.get(url)
                print(f"Generated URL: {url}")
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                return None


class Convert(BaseModel):
    model: Optional[list] = Field(default=None)
    datasets: Optional[list] = Field(default=None)
    keyword:Optional[str] = Field(default='id')
    list_key:Optional[list] = Field(default=[])
    model_amount:Optional[int] = Field(default=None)
    datasets_amount:Optional[int] = Field(default=None)
    


    def __init__(self, **data):
        super().__init__(**data)
        forbid_key = ['keyword','model_amount','datasets_amount']
        for key,value in data.items():
            if value is not None and key not in forbid_key:
                self.list_key.append(key)
                setattr(self,key,value)
        
    def convert_to_json(self):
        place_holder = {}
        datasets_array = []
        for key in self.list_key:
            for value in getattr(self,key):
                if key == 'model':
                    if len(place_holder) < self.model_amount:
                        place_holder['model'] = value[self.keyword]
                    pass
                elif key == 'datasets':
                    if len(datasets_array) < self.datasets_amount:
                        datasets_array.append(value[self.keyword])
                        place_holder['datasets'] = datasets_array
                    pass
    
        with open('DataModel_config/data_model.json','w') as f:
            json.dump(place_holder, f, indent=4)
 
        
        
        
        
# Example usage
if __name__ == "__main__":
    # Create an instance of APIFetch with only the parameters you want to use
    data_type = "models"
    search = "text-generation"
    model_api = APIFetch(
        web_address="https://huggingface.co/api/",
        type=data_type,
        search=search,
        model_name=["Orenguteng/Llama-3-8B-Lexi-Uncensored"],
    )
    
    all_model_name = model_api.get_api_json()
    
    data_type = "datasets"
    datasets_api = APIFetch(
        web_address="https://huggingface.co/api/",
        type=data_type,
        search=search
    )
    
    all_datasets_name = datasets_api.get_api_json()
    converter = Convert(model=all_model_name,datasets=all_datasets_name
                        ,keyword="id",model_amount=10,datasets_amount=10)
    converter.convert_to_json()
    

# WORKSPACE_DIR = Path(__file__).parent.parent.absolute()
# DATAMODEL_DIR = WORKSPACE_DIR / 'DataModel_config'

# DATAMODEL_DIR.mkdir(parents=True,exist_ok=True)

# class DatasetHandler:
#     def __init__(self, name,language,split):
#         self.name = name
#         self.language = language
#         self.split = split
#         self.dataset = load_dataset(name,
#                         language=self.language, 
#                         split=self.split,
#                         streaming=True,
#                         trust_remote_code=True) # optional, but the dataset only has a train split
        
        


#     def get_dataset(self):
#         return self.dataset


