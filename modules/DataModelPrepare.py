import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import torch
import numpy as np
from colorama import Fore, Style, init
from datasets import load_dataset, concatenate_datasets, DatasetDict,get_dataset_config_info,get_dataset_split_names,get_dataset_config_names
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    AutoPeftModelForCausalLM,
    prepare_model_for_kbit_training
)
# from trl import SFTTrainer
import evaluate
from huggingface_hub import HfApi

from modules.chatTemplate import ChatTemplate
# from modules.chainpipe import Chainpipe
from modules.createbasemodel import load_saved_model, CreateModel, VisionConfig, VisionModel

from modules.variable import Variable

import pandas as pd

# Initialize colorama
init(autoreset=True)

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1' 
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


    
    
class Manager:
    """Manager class for handling fine-tuning operations."""
    
    def __init__(self):
        self.variable = Variable()        
        self.repository = self.variable.REPO_DIR
        self.CUTOM_MODEL_DIR = self.variable.CUTOM_MODEL_DIR
        self.VISION_MODEL_DIR = self.variable.VISION_MODEL_DIR
        self.REGULAR_MODEL_DIR = self.variable.REGULAR_MODEL_DIR
        self.MODEL_LOCAL_DIR = self.variable.REPO_DIR
        self.training_config_path = self.variable.training_config_path
        self.DATASET_FORMATTED_DIR = self.variable.DATASET_FORMATTED_DIR
        self.device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.chat_template_saved = None
        self.chat_template_path = self.variable.chat_template_path
        os.makedirs(self.DATASET_FORMATTED_DIR, exist_ok=True)

    
    def get_model_architecture(self, model_id: str) -> List[str]:
        try:
            config = AutoConfig.from_pretrained(model_id)
            model_type = config.model_type.lower()
            
            target_modules_map = {
                "gpt2": ["c_attn", "c_proj"],
                "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "opt": ["q_proj", "k_proj", "v_proj", "out_proj"],
                "bloom": ["query_key_value", "dense"],
                "t5": ["q", "k", "v", "o"],
                "bert": ["query", "key", "value", "output.dense"],
                "roberta": ["query", "key", "value", "output.dense"],
                "gpt_neox": ["query_key_value", "dense"],
                "falcon": ["query_key_value", "dense"],
                "mpt": ["Wqkv", "out_proj"],
                "baichuan": ["W_pack", "o_proj"],
                "chatglm": ["query_key_value", "dense"],
                "qwen": ["c_attn", "c_proj"],
                "phi": ["Wqkv", "out_proj"],
                "gemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "stablelm": ["q_proj", "k_proj", "v_proj", "o_proj"]
            }
            
            target_modules = target_modules_map.get(model_type, ["q_proj", "k_proj", "v_proj", "o_proj"])
            
            print(f"{Fore.CYAN}Detected model architecture: {model_type}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Using target modules: {target_modules}{Style.RESET_ALL}")
            
            return target_modules
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not detect model architecture, using default target modules: {str(e)}{Style.RESET_ALL}")
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def get_model_task(self, model_name: str) -> str:
        try:
            api = HfApi()
            models = api.list_models(search=model_name)
            for model in models:
                if model.id.startswith(model_name):
                    return model.pipeline_tag
            return "text-generation"
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not determine model task, using default: {str(e)}{Style.RESET_ALL}")
            return "text-generation"
    

    def load_model(self, model_id: str) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
       
        print(f"{Fore.CYAN}Retrieving model {model_id}{Style.RESET_ALL}")

        
        try:

            self.model_task = self.get_model_task(model_id)
            print(f"{Fore.CYAN}Model task detected: {self.model_task}{Style.RESET_ALL}")
            
            return self._load_from_scratch(model_id)


        except Exception as e:
            print(f"{Fore.RED}Error loading model {model_id}: {str(e)}{Style.RESET_ALL}")
            return None, None
    

    def _load_from_scratch(self, model_id: str) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        model_path = self.variable.LocalModel_DIR / model_id
        try:
            print(f"{Fore.CYAN}Downloading and loading model: {model_path}{Style.RESET_ALL}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="right",
                truncation_side="right",
                
            )
            
            
            config = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_cache=False  # Disable cache for gradient checkpointing compatibility
            )
            
            # Get target modules for LoRA based on model architecture
            target_modules = self.get_model_architecture(model_id)
            
            # Configure quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=False
            )
            
            # Load base model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                device_map=self.device_map,
                trust_remote_code=True,
                quantization_config=bnb_config,
                torch_dtype=torch.float32,
            )
            
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=8,  # Rank
                lora_alpha=8,  # Alpha scaling
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Get PEFT model
            model = get_peft_model(model, lora_config)
            
            # Enable training mode and gradient checkpointing
            model.config.use_cache = False  # Ensure config is consistent
            model.train()
            model.gradient_checkpointing_enable()
            
            print(f"{Fore.GREEN}Successfully loaded model and tokenizer with LoRA configuration{Style.RESET_ALL}")
            return model, tokenizer
            
        except Exception as e:
            print(f"{Fore.RED}Error loading model from scratch: {str(e)}{Style.RESET_ALL} from {model_path}")
            return None, None
    
    def load_dataset(self, dataset_name: str, config: Optional[Dict] = None) -> Optional[DatasetDict]:
        
        print(f"{Fore.CYAN}Retrieving dataset {dataset_name}{Style.RESET_ALL}")
        
        valid_sep = ['train' if 'train' in get_dataset_split_names(dataset_name, config) else 'test' for config in get_dataset_config_names(dataset_name)]
        split = valid_sep[0] if valid_sep else 'train'
        
        
            
        try:
            self.dataset_name = dataset_name
            
            if config is not None:
                try:
                    print(f"{Fore.YELLOW}Attempting to load dataset with config: {config}{Style.RESET_ALL}")
                    dataset = load_dataset(dataset_name, config, split=split)
                    print(f"{Fore.GREEN}Successfully loaded dataset with config{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error loading dataset with config: {str(e)}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Trying to load dataset without config...{Style.RESET_ALL}")
                    try:
                        dataset = load_dataset(dataset_name, split=split)
                        print(f"{Fore.GREEN}Successfully loaded dataset without config{Style.RESET_ALL}")
                    except Exception as e2:
                        print(f"{Fore.RED}Error loading dataset without config: {str(e2)}{Style.RESET_ALL}")
                        return None
            else:
                try:
                    dataset = load_dataset(dataset_name, split=split)
                    print(f"{Fore.GREEN}Successfully loaded dataset without config{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error loading dataset: {str(e)}{Style.RESET_ALL}")
                    return None
            return dataset
        except Exception as e:
            print(f"{Fore.RED}Unexpected error loading dataset {dataset_name}: {str(e)}{Style.RESET_ALL}")
            return None
    
    def map_tokenizer(self, dataset_name: str, model_name: str, tokenizer: AutoTokenizer, dataset: DatasetDict, 
                     max_length: int = 1000, Tokenizing: bool = False) -> Optional[DatasetDict]:

        print(f"{Fore.CYAN}Processing dataset with max length: {max_length}{Style.RESET_ALL}")
        
        # Ensure tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"{Fore.GREEN}Set padding token to EOS token{Style.RESET_ALL}")
            
        self.chat_template = ChatTemplate(tokenizer=tokenizer,model_name=model_name)
        self.chat_template_saved = self.chat_template.tokenizer.chat_template
        try:
            tokenized_dataset = self.chat_template.prepare_dataset(
                dataset_name,
                dataset,
                max_length=max_length,
                Tokenizing=Tokenizing
            )
            print(f"{Fore.GREEN}Successfully prepared chat dataset{Style.RESET_ALL}")
            return tokenized_dataset
        except Exception as e:
            print(f"{Fore.RED}Error tokenizing dataset: {str(e)}{Style.RESET_ALL}")
            return None
    def dataset_prepare(self, list_model_data: List[Dict[str, Any]]) -> Tuple[Optional[AutoModelForCausalLM], Optional[DatasetDict]]:
        
        datamodel_file = self.variable.SAVED_CONFIG_Path
        
        datamodel_file = datamodel_file.as_posix()
        
        self.training_config_path.touch(exist_ok=True)
        
        
        try:
            with open(datamodel_file, 'r') as f:
                config = json.load(f)
        except:
            print(f"error config file not found {datamodel_file}")
            
        try:
            # combined_dataset = None
            dataset = None
            saved_dataset = None
            print(list_model_data)
            
            model_training_data = {'model':dict()}
            
            #load model and dataset prepare for tuning
            for modelname,dict_dataset in list_model_data['model'].items():

                model, tokenizer = self.load_model(modelname)

                union_cols = None
                saved_dataset = None
                first_dataset = None
                second_dataset = None
                
                first_cols = set()
                second_cols = set()
                concat_dataset = set()

                model_training_data['model'][modelname] = dict()

                
                for dataset_name,info in dict_dataset.items():
                    
                    formatted_dataset_name = f"{dataset_name.replace('/', '_')}_formatted"
                    
                    if not (self.DATASET_FORMATTED_DIR / formatted_dataset_name).exists():
                        print(f"{Fore.CYAN}Formatting Dataset {dataset_name}{Style.RESET_ALL}")
                        try:
                            print(f"{Fore.CYAN}Loading dataset config: {dataset_name} {config.get(dataset_name, 'No config found')}{Style.RESET_ALL}")
                            
                            dataset = self.load_dataset(dataset_name, config.get(dataset_name, 'default'))
                        
                            
                            if first_dataset is None:
                                print(f"{Fore.GREEN}Processing first dataset: {dataset_name}{Style.RESET_ALL}")
                                
                                #return processed True make it return text
                                first_dataset = self.map_tokenizer(dataset_name,
                                                                   modelname,
                                                                    tokenizer, dataset, 
                                                                    Tokenizing=False)
                                if first_dataset is None:
                                    print(f"{Fore.RED}Failed to process first dataset: {dataset_name}{Style.RESET_ALL}")
                                    continue
                                first_cols = set(first_dataset.column_names)
                                concat_dataset = first_dataset
                                
                                
                                            
                            else:
                                # first_dataset = concat_dataset
                                print(f"{Fore.GREEN}Processing additional dataset: {dataset_name}{Style.RESET_ALL}")
                                
                                #return processed True make it return text
                                second_dataset = self.map_tokenizer(dataset_name,
                                                                    modelname,
                                                                    tokenizer, 
                                                                    dataset, 
                                                                    Tokenizing=False)

                                concat_dataset = second_dataset
                                if second_dataset is None:
                                    print(f"{Fore.RED}Failed to process second dataset: {dataset_name}{Style.RESET_ALL}")
                                    continue

                                second_cols = set(second_dataset.column_names)
                                
                                
                                print(f"{Fore.GREEN}Concatenating datasets...{Style.RESET_ALL}")
                            
                                print(f"{Fore.GREEN}First dataset columns: {first_cols}{Style.RESET_ALL}")
                                print(f"{Fore.GREEN}Second dataset columns: {second_cols}{Style.RESET_ALL}")
                                                                
                            union_cols = first_cols.union(second_cols)
                            model_training_data['model'][modelname][dataset_name] = str(union_cols)
                            #after formatted to right format it use it to embedding
                            #after getting concatenate dataset return it to embedding formatted with return both false since the model going to tokenized it anyways
                            saved_dataset = self.map_tokenizer(dataset_name,
                                                               modelname,
                                                                tokenizer, 
                                                                concat_dataset,
                                                                Tokenizing=True)
                            
                            os.makedirs(self.DATASET_FORMATTED_DIR  / formatted_dataset_name, exist_ok=True)
                            saved_dataset.save_to_disk(self.DATASET_FORMATTED_DIR / formatted_dataset_name)
                            
                            if first_dataset is not None and second_dataset is not None:
                                # For columns only in second dataset, add them to first dataset with None values
                                for col in second_cols - first_cols:
                                    first_dataset = first_dataset.add_column(col, [None] * len(first_dataset))
                                
                                # For columns only in first dataset, add them to second dataset with None values  
                                for col in first_cols - second_cols:
                                    second_dataset = second_dataset.add_column(col, [None] * len(second_dataset))
                                
                                # Now both datasets have same columns, concatenate them
                                concat_dataset = concatenate_datasets([first_dataset, second_dataset])
                                print(f"{Fore.GREEN}Successfully joined datasets with columns: {concat_dataset.column_names}{Style.RESET_ALL}")
                                formatted_dataset_name = f"concat_{formatted_dataset_name}"
                                #after formatted to right format it use it to embedding
                                #after getting concatenate dataset return it to embedding formatted with return both false since the model going to tokenized it anyways
                                saved_dataset = self.map_tokenizer(dataset_name, 
                                                                    tokenizer,
                                                                    modelname,
                                                                    concat_dataset,
                                                                                Tokenizing=True)
                                
                                
                                os.makedirs(self.DATASET_FORMATTED_DIR  / formatted_dataset_name, exist_ok=True)
                                saved_dataset.save_to_disk(self.DATASET_FORMATTED_DIR / formatted_dataset_name)

                        except Exception as e:
                            print(f"{Fore.RED}Error processing dataset {dataset_name}: {str(e)}{Style.RESET_ALL}")
                            continue
                    else:
                        print(f"{Fore.GREEN}Formatted dataset already exists: {formatted_dataset_name}, loading...{Style.RESET_ALL}")
                        continue
                        
                if "conversations" in union_cols:
                    #if model is not local and been createdd
                    model_name_safe = modelname.replace("/","_")
                    model_path = self.REGULAR_MODEL_DIR / model_name_safe

                    if not (model_path).exists():
                        print(f"{Fore.GREEN}Creating conversation model...from {modelname}{Style.RESET_ALL}")
                        create_model = CreateModel(modelname, "conversation-model")
                        create_model.add_conversation()
                        create_model.save_regular_model()
                        with open(self.chat_template_path / "chat_template_conversation.jinja", 'w') as f:
                            f.write(self.chat_template_saved)

                #temporal fix this
                if "image" in union_cols or "images" in union_cols:
                    model_name_safe = modelname.replace("/","_")

                    model_path = self.VISION_MODEL_DIR / model_name_safe                       

                    if not (model_path).exists():
                        print(f"{Fore.GREEN}Creating vision model...from {modelname}{Style.RESET_ALL}")
                        create_model = CreateModel(modelname, "vision-model")
                        create_model.add_vision()
                        create_model.save_vision_model()
                        with open(self.chat_template_path / "chat_template_vision.jinja", 'w') as f:
                            f.write(self.chat_template_saved)

        

                   

            with open(self.training_config_path, 'w') as f:
                json.dump(model_training_data, f, indent=4)
            
        except Exception as e:
            print(f"{Fore.RED}Error running finetune: {str(e)}{Style.RESET_ALL}")
            return None, None
