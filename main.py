import os
from modules.variable import Variable
from modules.prerun import create_config_folders
from modules.ApiDump import ApiCardSetup
from modules.DataDownload import DataLoader

from modules.DataModelPrepare import Manager
from modules.inference import InferenceManager
from modules.train import FinetuneModel

from pathlib import Path


variable = Variable()
downloading = DataLoader()

# #finetune model
finetune = Manager()
Ft = FinetuneModel()

api = variable.hf_api


create_config_folders()

setcard = ApiCardSetup()

#set lists of models and datasets in config files
list_models = api.list_models(model_name='Qwen/Qwen1.5-0.5B-Chat',limit=1,gated=False)
list_datasets = api.list_datasets(dataset_name='laolao77/MMDU',limit=1,gated=False)

#set new list to download
list_download = setcard.set(list_models,list_datasets)

#download from datacard
downloading.run(list_download)
#formatting dataset
finetune.dataset_prepare(list_download)
# use formatted dataset and models
Ft.finetune_model()



# model_path = Path(__file__).parent.absolute() / "checkpoints" / "text-vision-text-generation" / "Qwen_Qwen1.5-0.5B-Chat"
# inference_manager = InferenceManager(model_path)
# image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcStFEnMosKZV8Y5Dy23L_kjxg7Jup75XA3Cpg&s"

# user_input = ""

# response = inference_manager.generate_response(user_input,image_path=image_path)
# print(f"{response}")

# image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRngDpJnVUWLRj3Vs5HEmhuzCgK0w5SEN0Mgg&s"
# user_input = ""
# response = inference_manager.generate_response(user_input,image_path=image_path)
# print(f"{response}")
