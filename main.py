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

# access_token = os.environ.get("hf_token")

# api = HfApi()
# variable = Variable()
# DMConfig_DIR = variable.DMConfig_DIR

# DMConfig_DIR.mkdir(parents=True, exist_ok=True)
# SavedConfigsfile= variable.SAVED_CONFIG_Path
# SavedConfigsfile.touch(exist_ok=True)

# setcard = ApiCardSetup()

# list_models = api.list_models(model_name='Qwen/Qwen1.5-0.5B-Chat',limit=1,gated=False)
# # list_models = api.list_models(model_name='Qwen/Qwen3-4B-Instruct-2507',limit=1,gated=False)
# # list_datasets = api.list_datasets(dataset_name='QHQK/conversation_hall_binary_v1',limit=1,gated=False)

# # list_datasets = api.list_datasets(dataset_name='ThucPD/coco-qa-vi',limit=1,gated=False)
# list_datasets = api.list_datasets(dataset_name='waltsun/MOAT',limit=1,gated=False)
# # list_datasets = api.list_datasets(dataset_name='Chappieut/MarvelRivals10Heroes',limit=1,gated=False)
# # list_datasets = api.list_datasets(dataset_name="FreedomIntelligence/medical-o1-reasoning-SFT",limit=1,gated=False)


# #set new list to download
# list_download = setcard.set(list_models,list_datasets)

# downloading = DataLoader()

# #download from datacard
# downloading.run(list_download)


# # #finetune model
# finetune = Manager()

# finetune.dataset_prepare(list_download)

Ft = FinetuneModel()
Ft.finetune_model()




# Inference setup
# print("\n--- Starting Inference ---\n")
# model_path = Path(__file__).parent.absolute() / "checkpoints" / "text-generation" / "Qwen_Qwen1.5-0.5B-Chat"
# model_path = Path(__file__).parent.absolute() / "repositories" / "models" / "Qwen" / "Qwen1.5-0.5B-Chat"
# model_path = Path(__file__).parent.absolute() / "checkpoints" / "text-vision-text-generation" / "Qwen_Qwen1.5-0.5B-Chat"
# model_path = Path(__file__).parent.absolute() / "custom_models" / "conversation-model" / "Qwen_Qwen1.5-0.5B-Chat"
# model_path = Path(__file__).parent.absolute() / "custom_models" / "vision-model" / "Qwen_Qwen1.5-0.5B-Chat" / "lang_model"
# inference_manager = InferenceManager(model_path)

# # # Example 1: Text-only inference
# user_input = "who is spider gwen?"

# # # # # # Print the formatted chat
# response = inference_manager.generate_response(user_input)
# print(f"{response}")

# Example 2: Multimodal inference
# model_path = Path(__file__).parent.absolute() / "custom_models" / "vision-model" / "Qwen_Qwen1.5-0.5B-Chat"
# inference_manager = InferenceManager(model_path)

# # image_path = "https://media.istockphoto.com/id/155439315/photo/passenger-airplane-flying-above-clouds-during-sunset.jpg?s=612x612&w=0&k=20&c=LJWadbs3B-jSGJBVy9s0f8gZMHi2NvWFXa3VJ2lFcL0="  # Replace with the actual path to your image
# image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThZJGbCDib3lJ4yPnBUSHgFawk_heC84NxGA&s"
# user_input = "who is spidergwen? What is she doing in this image?"
# response = inference_manager.generate_response(user_input, image_path=image_path)
# print(f"{response}")
