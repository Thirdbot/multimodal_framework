#use this to download the data from the data_model.json and successully save to installed.json and sent model to finetuned for use again in main.py
import pandas as pd 
import json
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import hf_hub_download,HfApi
from pathlib import Path
from DatasetHandler import APIFetch,Convert
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class FlexibleDatasetLoader:
    def __init__(self, config=None, split='train',trust_remote_code=True):
        self.trust_remote_code = trust_remote_code
        self.config = config
        self.split = split
        # self.dataset = self._load(self.config)
        
    def load(self,name):
        
        if self.config:
            return load_dataset(name, self.config, split=self.split,trust_remote_code=self.trust_remote_code)
        else:
            configs = get_dataset_config_names(name,trust_remote_code=self.trust_remote_code)
            if isinstance(configs,list):
                print(configs)
                user_input = input("Enter the config you want to use: ")
                self.config = user_input
                self.load(name)

    def get(self):
        return self.dataset
    
# dataset = FlexibleDatasetLoader(name="oscar-corpus/OSCAR-2201")
# print(dataset.get())


class ModelLoader:
    def __init__(self):
        super().__init__()
        self.file_paths = {}
        self.api = HfApi()
    def load_model(self,name):
       
        files = self.api.list_repo_files(name)
        for file_info in files:
            file_name = file_info.rsplit("/", 1)[-1]  # Extract filename
            file_path = hf_hub_download(
                repo_id=name,
                filename=file_name
            )
            self.file_paths[file_name] = file_path
            print(f"{file_name} downloaded to: {self.file_paths[file_name]}")
            
            
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
        
        self.datamodel = self.datamodel_load()
        self.load(self.datamodel)
    
    def datainstall_load(self):
        if self.installed_filepath.exists():
            if self.installed_filepath.stat().st_size > 0:
                installed = pd.read_json(self.installed_filepath)
                return installed
            else:
                return pd.DataFrame(columns=['model','datasets'])

    #load the data_model.json  if not hav a data then create a new one
    def datamodel_load(self):
         if self.datamodel_filepath.exists():
            if self.datamodel_filepath.stat().st_size > 0:
                df = pd.read_json(self.datamodel_filepath)
                return df
            else:

                model_api = APIFetch(
                            web_address="https://huggingface.co/api/",
                            type=self.data_type,
                            task_categories=['text-generation','image-to-text'],
                            # author="huggingface",
                            # model_name=["Orenguteng/Llama-3-8B-Lexi-Uncensored"
                            #             ,"nari-labs/Dia-1.6B"],
                            # datasets_name=[["nvidia/OpenMathReasoning",
                            #                "Anthropic/values-in-the-wild"],
                            #                ['nvidia/describe-anything-dataset',
                            #                 ]]
                )
            
                all_model_name = model_api.get_api_json()


                # all_datasets_name = datasets_api.get_api_json()
                converter = Convert(data_model=all_model_name
                                    ,keyword="id",model_amount=10,datasets_amount=10)
                return converter.convert_to_json(self.datamodel_filepath)


  
                
    def load(self,install, depth=0):
        installed = []
        datadict = {'model':str,'datasets':[]}
        base_df = pd.DataFrame(install)
        compare_df = pd.DataFrame(self.datainstall_load())
        
        if depth > 3:
            print("Maximum retry depth reached.")
            return
        if base_df.equals(compare_df):
                print("Data already installed")
        else:
        
           base_df['datasets'] = base_df['datasets'].apply(lambda x: sorted(x))
           compare_df['datasets'] = compare_df['datasets'].apply(lambda x: sorted(x))
            # Convert to Series with names before merging
           base_df['__row__'] = base_df.apply(lambda row: json.dumps(row.to_dict(), sort_keys=True), axis=1)
           compare_df['__row__'] = compare_df.apply(lambda row: json.dumps(row.to_dict(), sort_keys=True), axis=1)
            
      
           diff_model = base_df[~base_df['__row__'].isin(compare_df['__row__'])]
           
           
           diff_model = diff_model.drop(columns='__row__').to_dict(orient='records')
                
           diff_model = pd.DataFrame(diff_model)

           if diff_model.empty:
                print("No new models to install")
                return
           failed_models = []

           for i, row in diff_model.iterrows():
                try:
                    model = row['model']
                    datasets = row['datasets']
                    print(model,datasets)
                    # self.model.load_model(model)
                    if isinstance(datasets,list):
                        for dataset in datasets:
                            pass
                            # self.dataset.load(dataset)
                    else:
                        # self.dataset.load(datasets)
                        pass
                    
                    datadict = {
                        'model': model,
                        'datasets': datasets
                    }

                    installed.append(datadict)
                    # insert actual install logic here if needed
                    

                except:
                    pass
           if installed:
                prev_df = pd.DataFrame(self.datainstall_load())
                new_installs_df = pd.DataFrame(installed)
                result_df = pd.concat([prev_df, new_installs_df], ignore_index=True)
                with open(self.installed_filepath, 'w') as f:
                    json.dump(result_df.to_dict(orient='records'), f, indent=4)

           if failed_models:
                print(f"Retrying {len(failed_models)} failed installs...")
                self.load(failed_models, depth=depth + 1)
                
                
                    
            
               


if __name__ == "__main__":
    loader = DataLoader()
    
   
            
    