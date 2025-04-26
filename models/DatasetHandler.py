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
    type: Optional[list] = Field(default=None)
    search: Optional[str] = Field(default=None)
    author: Optional[str] = Field(default=None)
    task_categories: Optional[list] = Field(default=[])
    tag: Optional[str] = Field(default=None)
    list_key:Optional[list] = Field(default=[])
    
    model_name:Optional[list] = Field(default=[])
    datasets_name:Optional[list] = Field(default=[])

    
    
    def __init__(self, **data):
        super().__init__(**data)
        forbid_key = ['model_name','type','datasets_name','web_address','task_categories']
        # Only set attributes that were actually passed in
        for key, value in data.items():
            if value is not None and key not in forbid_key:
                self.list_key.append(key)
            setattr(self,key,value)
                
                
   

    def concatenate_request(self):
        
        # all_url = []
        target_url = []
        data_list = []
        model_data_url = {'model':str,'datasets':[]}
        if not self.web_address:
            return None
        
        
        # web_address = self.web_address + type
        params = {}
        for type in self.type:
           
            web_address = self.web_address.rstrip(',')
            web_address = web_address + type
            # list of url target model

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
                            web_address = web_address + '/' + datasets_name
                            
                            data_list.append(web_address)
                            
                            web_address = self.web_address + type
                    # model_data_url['datasets'] = data_list
                
                        target_url[idx]['datasets'] = data_list
                        data_list = []
                    
                    if len(target_url) > 1:
                        return target_url
                else:
                    for datasets_name in self.datasets_name:
                        web_address = web_address + '/' + datasets_name
                        
                        data_list.append(web_address)
                        
                        web_address = self.web_address + type
                    target_url = [{**trt, 'datasets': data_list} for trt in target_url]
                    if len(target_url) > 1:
                        
  
                        return target_url
           
            # Get all non-None attributes that were actually set
            
            
            for key in self.list_key:
                for value in getattr(self,key):  # Only include if it was actually set
                    if value is not None:
                        params[key] = value
            
            # Construct URL with parameters
            # if params:
            print(web_address)
            param_string = '&'.join(f"{k}={v}" for k, v in params.items() 
                                    if not(isinstance(pv, list) for pv in params['task_categories']))
            print(param_string)
            if len(self.task_categories) >= 1:
        
                for idx ,task in enumerate(self.task_categories ):
                        data = f"{web_address}?{param_string}&task_categories={task}"
                        model_data_url['model'] = data
                        target_url.append(model_data_url)
                        model_data_url = {'model':str,'datasets':[]}
                        data_list.append(data)
                        target_url[idx]['datasets'] = data_list
                        data_list = []
                return target_url
     
                    
       
    def get_api_json(self):
        concatenated_json = []
        data_list = []
        data_model_json = {"model":str,"datasets":[]}
        url = self.concatenate_request()
        print(url)
        
        if not url:
            return None
        if isinstance(url,list):
            for url in url:
                if not isinstance(url,dict):
                    try:
                        response = requests.get(url)
                        print(f"Generated URL: {url}")
                        response.raise_for_status()
                        concatenated_json.append(response.json())
                    except requests.exceptions.RequestException as e:
                        print(f"Error fetching data: {e}")
                        continue
                else:
                    try:
                        response = requests.get(url['model'])
                        print(f"Generated MODEL URL: {url['model']}")
                        response.raise_for_status()
                        data_model_json['model'] = response.json()
                        concatenated_json.append(data_model_json)
                    except requests.exceptions.RequestException as e:
                        print(f"Error fetching data: {e}")
                        continue
                    for datasets in url['datasets']:
                        try:
                            response = requests.get(datasets)
                            print(f"Generated DATASETS URL: {datasets}")
                            response.raise_for_status()
                            data_list.append(response.json())
                        
                        
                        except requests.exceptions.RequestException as e:
                            print(f"Error fetching data: {e}")
                            continue
                    concatenated_json = [{**cj, 'datasets': data_list} for cj in concatenated_json]
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
        name_list = []
        place_holder = {'model':str,'datasets':[]}
        datasets_array = []
        i = 0
        for key in self.list_key:
            for value in getattr(self,key):
                if key == 'model':
                    print(value[self.keyword])
                    if len(name_list) < self.model_amount:
                        print(value[self.keyword])
                        place_holder['model'] = value[self.keyword]
                        name_list.append(place_holder)
                    
                    pass
                elif key == 'datasets':
                    print(name_list)
                    if len(name_list[i]['datasets']) < self.datasets_amount:
                        datasets_array.append(value[self.keyword])
                        name_list[i]['datasets'] = datasets_array
                    pass
                    i += 1
        with open('DataModel_config/data_model.json','w') as f:
            json.dump(name_list, f, indent=4)
 
        
        
        
        
# Example usage
if __name__ == "__main__":
    # Create an instance of APIFetch with only the parameters you want to use
    data_type = ["models","datasets"]
    # task = "text-generation"
    model_api = APIFetch(
        web_address="https://huggingface.co/api/",
        type=data_type,
        # task_categories=['text-generation','text-classification'],
        model_name=["Orenguteng/Llama-3-8B-Lexi-Uncensored"
                    ,"nari-labs/Dia-1.6B"],
        datasets_name=[["nvidia/OpenMathReasoning",
                       "Anthropic/values-in-the-wild"],
                       ['nvidia/describe-anything-dataset']]
    )
    
    all_model_name = model_api.get_api_json()
    
    # data_type = "datasets"
    # datasets_api = APIFetch(
    #     web_address="https://huggingface.co/api/",
    #     type=data_type,
    #     search=search
    # )
    
    # all_datasets_name = datasets_api.get_api_json()
    # converter = Convert(model=all_model_name,datasets=all_datasets_name
    #                     ,keyword="id",model_amount=10,datasets_amount=10)
    # converter.convert_to_json()
    

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


