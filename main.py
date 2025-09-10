
# # What To do

# # 2. redesign model-dataset relation loop
# # 3. use vllm for model inference and still huggingface compatible for model push
# # 4. be able to update new repository as a docker container for run on big gpu
# # 5. each file independent and can be run with argument

# #make model seperate tokenizer and model
# #renew interference utilize langchain model
# # #try cut conner of chat template and create or utilize model's tokenizer to make dataste compatible with model include add model's compatible pipeline for multimodal


# #handle various of datasets downloadble files need each column
# #create template dataset
# #auto fallback dataset load request when failed to tokenize a dataset in case of naming convension

# #this just addon.
# # create api for model
# #app for multimodal

# #main stuff
# # create multimodal template
# #merge dataset to multimodal
# #create event loop for use input and model input (should recieve multiple input type data as sametime)
# #create attention between event loop for filter unwanting data so runtime not interfere
# #create embbeding and en-router-attention with de-router-attention (shared attention or embed)
# #create function feed input from router to encoder_model or decoder model
# #create function to display output by model output from router_attention
# #####note output from model should be stream into input of model_input instead of user_input or its model input for inteferencing
# #####the data should be on eventloop instead of model loop so crack that 1 bit llms


#damn life.... i want a contributor


import os
from huggingface_hub import HfApi
# from datasets import load_dataset,get_dataset_split_names
from modules.ApiDump import ApiCardSetup
from modules.DataDownload import DataLoader
# from modules.finetuning_model import FinetuneModel
from modules.DataModelPrepare import Manager
from modules.variable import Variable
from modules.inference import InferenceManager
from modules.train import FinetuneModel
from pathlib import Path

acess_token = os.environ.get("hf_token")

api = HfApi()
variable = Variable()
DMConfig_DIR = variable.DMConfig_DIR

DMConfig_DIR.mkdir(parents=True, exist_ok=True)
SavedConfigsfile= variable.SAVED_CONFIG_Path
SavedConfigsfile.touch(exist_ok=True)

setcard = ApiCardSetup()

list_models = api.list_models(model_name='Qwen/Qwen1.5-0.5B-Chat',limit=1,gated=False)
list_datasets = api.list_datasets(dataset_name='waltsun/MOAT',limit=1,gated=False)
# list_datasets = api.list_datasets(dataset_name="FreedomIntelligence/medical-o1-reasoning-SFT",limit=1,gated=False)


#set new list to download
list_download = setcard.set(list_models,list_datasets)

downloading = DataLoader()

#download from datacard
downloading.run(list_download)


# #finetune model
finetune = Manager()

finetune.dataset_prepare(list_download)

Ft = FinetuneModel()
Ft.finetune_model()




# # Inference setup
# # print("\n--- Starting Inference ---\n")
# model_path = Path(__file__).parent.absolute() / "checkpoints" / "text-generation" / "Qwen_Qwen1.5-0.5B-Chat"
# # model_path = Path(__file__).parent.absolute() / "repositories" / "models" / "Qwen" / "Qwen1.5-0.5B-Chat"
model_path = Path(__file__).parent.absolute() / "checkpoints" / "text-vision-text-generation" / "Qwen_Qwen1.5-0.5B-Chat"
# # model_path = Path(__file__).parent.absolute() / "custom_models" / "conversation-model" / "Qwen_Qwen1.5-0.5B-Chat"
# inference_manager = InferenceManager(model_path)

# # # Example 1: Text-only inference
# user_input = "who is spider gwen?"

# # Print the formatted chat
# response = inference_manager.generate_response(user_input)
# print(f"{response}")

# # # # Example 2: Multimodal inference
# model_path = Path(__file__).parent.absolute() / "custom_models" / "vision-model" / "Qwen_Qwen1.5-0.5B-Chat"
inference_manager = InferenceManager(model_path)

image_path = "https://media.istockphoto.com/id/155439315/photo/passenger-airplane-flying-above-clouds-during-sunset.jpg?s=612x612&w=0&k=20&c=LJWadbs3B-jSGJBVy9s0f8gZMHi2NvWFXa3VJ2lFcL0="  # Replace with the actual path to your image
user_input = "what is this images about? can you describe it to me?"
response = inference_manager.generate_response(user_input, image_path=image_path)
print(f"{response}")













#left to do
# vllm model inference
# train in seperated file
# doccker all dependence files like embedded_dataset model train script and etc..
# implement runpod to manage training jobs from docker
# after concatenate dataset it should merge/rearrange the dataset for training in multimodality
#should fix chat template to format using downloaded model jinja template as text formatter


