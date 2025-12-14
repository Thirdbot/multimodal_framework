import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple

import torch
from colorama import Fore, Style, init
from datasets import load_dataset, concatenate_datasets, DatasetDict, get_dataset_split_names, get_dataset_config_names
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from modules.chatTemplate import ChatTemplate
from modules.ModelUtils import CreateModel
from modules.variable import Variable

# Initialize colorama
init(autoreset=True)

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1' 
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


@dataclass
class ManagerConfig:
    device: Optional[str] = None  # e.g., "cuda:0" or "cpu"
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    load_in_4bit: bool = True
    bnb_compute_dtype: torch.dtype = torch.float32
    bnb_quant_type: str = "fp4"
    bnb_use_double_quant: bool = False
    max_length: int = 1000


    
    
class Manager:
    """Manager class for handling fine-tuning operations."""
    
    def __init__(self, config: ManagerConfig | None = None):
        self.variable = Variable()
        self.config = config or ManagerConfig()
        self.VISION_MODEL_DIR = self.variable.VISION_MODEL_DIR
        self.REGULAR_MODEL_DIR = self.variable.REGULAR_MODEL_DIR
        self.training_config_path = self.variable.training_config_path
        self.DATASET_FORMATTED_DIR = self.variable.DATASET_FORMATTED_DIR
        self.chat_template_saved = None
        self.chat_template_path = self.variable.chat_template_path

        # Device selection
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device_map = self.config.device or default_device

        os.makedirs(self.DATASET_FORMATTED_DIR, exist_ok=True)
    
    def load_model(self, model_id: Union[str, Path]) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        print(f"{Fore.CYAN}Retrieving model {model_id}{Style.RESET_ALL}")

        try:
            return self._load_from_scratch(model_id)
        except Exception as e:
            print(f"{Fore.RED}Error loading model {model_id}: {str(e)}{Style.RESET_ALL}")
            return None, None
    

    def _load_from_scratch(self, model_id: Union[str, Path]) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        potential_path = Path(model_id)
        model_path = potential_path if potential_path.exists() else (self.variable.LocalModel_DIR / str(model_id))
        try:
            print(f"{Fore.CYAN}Downloading and loading model: {model_path}{Style.RESET_ALL}")

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="right",
                truncation_side="right",
            )

            # Ensure tokenizer has padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"{Fore.YELLOW}Set padding token to EOS token{Style.RESET_ALL}")

            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_cache=False  # Disable cache for gradient checkpointing compatibility
            )

            # Update config with tokenizer's pad_token_id
            if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                config.pad_token_id = tokenizer.pad_token_id

        
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                device_map=self.device_map,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )

            # Enable training mode and gradient checkpointing
            model.config.use_cache = False  # Ensure config is consistent
            model.train()
            model.gradient_checkpointing_enable()

            print(f"{Fore.GREEN}Successfully loaded model and tokenizer{Style.RESET_ALL}")
            return model, tokenizer
            
        except Exception as e:
            print(f"{Fore.RED}Error loading model from scratch: {str(e)}{Style.RESET_ALL} from {model_path}")
            return None, None
    
    def load_dataset(self, dataset_name: str, config_name: Optional[str] = None) -> Optional[DatasetDict]:
        print(f"{Fore.CYAN}Retrieving dataset {dataset_name}{Style.RESET_ALL}")

        try:
            splits = get_dataset_split_names(dataset_name, config_name)
            split = 'train' if 'train' in splits else ('test' if 'test' in splits else 'train')
        except Exception:
            split = 'train'

        try:
            if config_name is not None:
                print(f"{Fore.YELLOW}Attempting to load dataset with config: {config_name}{Style.RESET_ALL}")
                return load_dataset(dataset_name, config_name, split=split)

            print(f"{Fore.YELLOW}Loading dataset without config...{Style.RESET_ALL}")
            return load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"{Fore.RED}Error loading dataset {dataset_name}: {str(e)}{Style.RESET_ALL}")
            return None
    
    def map_tokenizer(self, dataset_name: str, model_name: str, tokenizer: AutoTokenizer, dataset: DatasetDict, 
                     max_length: Optional[int] = None, Tokenizing: bool = False) -> Optional[DatasetDict]:

        max_len = max_length or self.config.max_length
        print(f"{Fore.CYAN}Processing dataset with max length: {max_len}{Style.RESET_ALL}")
        
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
                max_length=max_len,
                Tokenizing=Tokenizing
            )
            print(f"{Fore.GREEN}Successfully prepared chat dataset{Style.RESET_ALL}")
            return tokenized_dataset
        except Exception as e:
            print(f"{Fore.RED}Error tokenizing dataset: {str(e)}{Style.RESET_ALL}")
            return None
    def dataset_prepare(self, list_model_data: Dict[str, Any]) -> Tuple[Optional[AutoModelForCausalLM], Optional[DatasetDict]]:
        
        datamodel_file = self.variable.SAVED_CONFIG_Path
        
        datamodel_file = datamodel_file.as_posix()
        
        self.training_config_path.touch(exist_ok=True)
        
        
        try:
            with open(datamodel_file, 'r') as f:
                config = json.load(f)
        except Exception:
            print(f"error config file not found {datamodel_file}")
            config = {}
            
        try:
            # combined_dataset = None
            dataset = None
            saved_dataset = None
            print(list_model_data)
            
            model_training_data = {'model':dict()}
            
            #load model and dataset prepare for tuning
            for modelname,dict_dataset in list_model_data.get('model', {}).items():

                model_repo = self.variable.REPO_DIR / "models" / modelname
                model, tokenizer = self.load_model(model_repo)

                union_cols = None
                saved_dataset = None
                first_dataset = None
                second_dataset = None
                
                first_cols = set()
                second_cols = set()
                concat_dataset = None

                model_training_data['model'][modelname] = dict()

                
                for dataset_name,info in dict_dataset.items():
                    
                    formatted_dataset_name = f"{dataset_name.replace('/', '_')}_formatted"
                    
                    print(f"{Fore.CYAN}Formatting Dataset {dataset_name}{Style.RESET_ALL}")
                    try:
                        print(f"{Fore.CYAN}Loading dataset config: {dataset_name} {config.get(dataset_name, 'No config found')}{Style.RESET_ALL}")
                        
                        dataset = self.load_dataset(dataset_name, config.get(dataset_name))
                    
                        
                        if first_dataset is None:
                            print(f"{Fore.GREEN}Processing first dataset: {dataset_name}{Style.RESET_ALL}")
                            
                            #return processed True make it return text
                            first_dataset = self.map_tokenizer(dataset_name,
                                                                model_repo,
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
                                                                model_repo,
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
                                                            model_repo,
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
                                                                model_repo,
                                                                tokenizer,
                                                                concat_dataset,
                                                                Tokenizing=True)
                            
                            
                            os.makedirs(self.DATASET_FORMATTED_DIR  / formatted_dataset_name, exist_ok=True)
                            saved_dataset.save_to_disk(self.DATASET_FORMATTED_DIR / formatted_dataset_name)

                    except Exception as e:
                        print(f"{Fore.RED}Error processing dataset {dataset_name}: {str(e)}{Style.RESET_ALL}")
                        continue

                   

            with open(self.training_config_path, 'w') as f:
                json.dump(model_training_data, f, indent=4)
            
        except Exception as e:
            print(f"{Fore.RED}Error running finetune: {str(e)}{Style.RESET_ALL}")
            return None, None
