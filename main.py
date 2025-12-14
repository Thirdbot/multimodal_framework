import os
from modules.variable import Variable
from modules.prerun import create_config_folders
from modules.ApiDump import ApiCardSetup
from modules.DataDownload import DataLoader

from modules.DataModelPrepare import Manager
from modules.inference import InferenceManager
from modules.train import FinetuneModel

from pathlib import Path

# if there is no custom_models folder, create it using create_model.py
# then run this file to download dataset and finetune the model
# after first run you can directly finetune without formatting dataset again by keeping FinetunModel.finetune_model

# variable = Variable()
# downloading = DataLoader()

# #finetune model
# finetune = Manager()
Ft = FinetuneModel()

# api = variable.hf_api


# create_config_folders()

# setcard = ApiCardSetup()

# # set lists of models and datasets in config files btw, set the name that not exist to make it skipped
# # new model add details in configs/ApiCardSet.json to make it works (for now)
# list_models = api.list_models(model_name='Qwen/Qwen1.5-0.5B-Chat',limit=1,gated=False)
# list_datasets = api.list_datasets(dataset_name='waltsun/MOAT',limit=1,gated=False)

# # set new list to download
# list_download = setcard.set(list_models,list_datasets)

# # download from datacard
# downloading.run(list_download)
# # formatting dataset

# ## use custom_models in custom_models folder so it need to have custom_models in folder 
# finetune.dataset_prepare(list_download)
# use formatted dataset and models
Ft.finetune_model()



# model_path = Path(__file__).parent.absolute() / "checkpoints" / "text-vision-text-generation" / "Qwen_Qwen1.5-0.5B-Chat"
# model_path = Path(__file__).parent.absolute() / "custom_models" / "conversation-model" / "newmodel"
# model_path = Path(__file__).parent.absolute() / "checkpoints" / "text-generation" / "Qwen_Qwen1.5-0.5B-Chat"

# inference_manager = InferenceManager(model_path)
# # image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSKrRTIhPqYvZTuh0m79LUYJsDRG9VgZYIaNA&s"

# # user_input = "What is the total value in the image?"

# # response = inference_manager.generate_response(user_input,image_path=image_path)
# # print(f"{response}")

# # image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRs0cV933dMsauoMHgQBpcZ-VTsTa5SbTAMWQ&s"
# user_input = "who is spidergwen?"
# response = inference_manager.generate_response(user_input)
# print(f"{response}")
