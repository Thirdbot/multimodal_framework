import os
import json
from colorama import Fore, Style, init
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from modules.defect import Report
from modules.chatTemplate import ChatTemplate
# Initialize colorama
init(autoreset=True)

# from langchain.llms import LlamaCpp
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from trl import SFTTrainer



# from peft.tuners.lora import mark_only_lora_as_trainable

from peft import LoraConfig, get_peft_model,PeftModel, AutoPeftModelForCausalLM, prepare_model_for_kbit_training

import torch

from pathlib import Path
import evaluate

import numpy as np
from transformers import  AutoModel,AutoTokenizer,AutoConfig,BitsAndBytesConfig,DataCollatorForLanguageModeling,AutoModelForCausalLM, Trainer, TrainingArguments,pipeline
from datasets import load_dataset, concatenate_datasets, DatasetDict
import re
from huggingface_hub import HfApi

# --fine-tune and merge with base-model through finetuning_model.py with custom multimodal embedding

# -- finetume with  any model that compatible with base-model architecture with autoclass to make same architecture from base-model



# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["PYTORCH_USE_CUDA_DSA"] = "1"



# # Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
#     temperature=0.75,
#     max_tokens=2000,
#     top_p=1,
#     callback_manager=callback_manager,
#     verbose=True,  # Verbose is required to pass to the callback manager
# )


class FinetuneModel:
    def __init__(self):
        # Training parameters
        self.per_device_train_batch_size = 32
        self.per_device_eval_batch_size = 32
        self.gradient_accumulation_steps = 2
        self.learning_rate = 2e-4
        self.num_train_epochs = 0.2
        self.save_strategy = "best"
        
        # Define paths
        self.WORKSPACE_DIR = Path(__file__).parent.parent.absolute()
        self.MODEL_DIR = self.WORKSPACE_DIR / "models"
        self.CHECKPOINT_DIR = self.WORKSPACE_DIR / "checkpoints"
        self.OFFLOAD_DIR = self.WORKSPACE_DIR / "offload"
        
        # Create directories
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        self.device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Set memory optimization environment variables
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.cuda.empty_cache()
        
        self.model_id = None
        self.dataset_name = None
        self.metric = evaluate.load("accuracy")
        self.model_task = None
        self.resume_from_checkpoint = True

        self.chat_template = ChatTemplate()
    
    def get_model_architecture(self, model_id):
        """Detect the model architecture and return appropriate LoRA configuration"""
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model_type = config.model_type.lower()
            
            # Common target modules for different architectures
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
            
            # Default to common attention patterns if architecture not found
            target_modules = target_modules_map.get(model_type, ["q_proj", "k_proj", "v_proj", "o_proj"])
            
            print(f"{Fore.CYAN}Detected model architecture: {model_type}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Using target modules: {target_modules}{Style.RESET_ALL}")
            
            return target_modules
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not detect model architecture, using default target modules: {str(e)}{Style.RESET_ALL}")
            return ["q_proj", "k_proj", "v_proj", "o_proj"]

    def get_model_task(self, model_name):
        try:
            api = HfApi()
            models = api.list_models(search=model_name)
            for model in models:
                if model.id.startswith(model_name):  # match closely
                    return model.pipeline_tag
            return "text-generation"  # default task if not found
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not determine model task, using default: {str(e)}{Style.RESET_ALL}")
            return "text-generation"

    def find_last_checkpoint(self, model_name):
        """Find the last checkpoint for a model"""
        try:
            model_task = self.get_model_task(model_name)
            model_name = model_name.replace('/', '_') if '/' in model_name else model_name
            
            checkpoint_dir = self.CHECKPOINT_DIR / model_task / model_name
            
            if not checkpoint_dir.exists():
                return None
            
            # Get all checkpoint directories
            checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
            
            if not checkpoints:
                return None
            
            # Sort by checkpoint number and get the latest
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
            return latest_checkpoint
            
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not find last checkpoint: {str(e)}{Style.RESET_ALL}")
            return None

    def load_model(self, model_id, resume_from_checkpoint=False):
        print(f"{Fore.CYAN}retrieve model {model_id}{Style.RESET_ALL}")
        
        self.resume_from_checkpoint = resume_from_checkpoint
        print(f"Load from last checkpoint at :{self.resume_from_checkpoint}")
        
        try:
            # Get model task
            self.model_task = self.get_model_task(model_id)
            print(f"{Fore.CYAN}Model task detected: {self.model_task}{Style.RESET_ALL}")
            
            # Create task-specific directories
            self.TASK_MODEL_DIR = self.MODEL_DIR / self.model_task
            self.TASK_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            
            # Find last checkpoint if resuming
            if resume_from_checkpoint:
                self.last_checkpoint = self.find_last_checkpoint(model_id)
                if self.last_checkpoint:
                    print(f"{Fore.CYAN}Resuming from checkpoint: {self.last_checkpoint}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}No checkpoint found, starting from scratch{Style.RESET_ALL}")
            
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                padding_side="right",
                truncation_side="right"
            )
            
            # Initialize ChatTemplate with the already loaded tokenizer
            self.chat_template = ChatTemplate(tokenizer=tokenizer)
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            
            # Load model config
            config = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                quantization_config=quantization_config,
                device_map=self.device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                offload_folder=str(self.OFFLOAD_DIR),
                offload_state_dict=True,
                max_memory={0: "40GB"}
            )
            
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True
            )
            
            # Get target modules
            target_modules = self.get_model_architecture(model_id)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Get PEFT model
            model = get_peft_model(model, lora_config)
            
            # Enable gradient checkpointing
            model.gradient_checkpointing_enable()
            
            # Print trainable parameters
            model.print_trainable_parameters()
            
            # Set padding token if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.pad_token_id
            
            return model, tokenizer
            
        except Exception as e:
            print(f"{Fore.RED}Error loading model {model_id}: {str(e)}{Style.RESET_ALL}")
            return None, None

    def load_dataset(self,dataset_name):
        print(f"{Fore.CYAN}retrieve dataset {dataset_name}{Style.RESET_ALL}")
        
        try:
            # Load dataset from Hugging Face
            self.dataset_name = dataset_name
            dataset = load_dataset(dataset_name,trust_remote_code=True)
            # print(Fore.YELLOW+"Dataset feature: "+str(dataset.features)+Fore.RESET)
            return dataset
        except Exception as e:
            print(f"{Fore.RED}Error loading dataset {dataset_name}: {str(e)}{Style.RESET_ALL}")
            return None
        
    def tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                padding_side="right",
                truncation_side="right"
            )
            # Set padding token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            print(f"{Fore.RED}Error loading tokenizer: {str(e)}{Style.RESET_ALL}")
            return None

    def map_tokenizer(self, tokenizer, dataset, max_length=384):
        try:
            # Get the first example to check available fields
            first_example = dataset["train"][0] if "train" in dataset else dataset[0]
            available_fields = list(first_example.keys())
            
            # Check if this is a chat/conversation dataset
            is_chat_dataset = "conversations" in available_fields or "messages" in available_fields
            
            if is_chat_dataset:
                print(f"{Fore.CYAN}Detected chat/conversation dataset{Style.RESET_ALL}")
                
                # Use ChatTemplate to prepare the dataset
                try:
                    tokenized_dataset = self.chat_template.prepare_dataset(
                        dataset,
                        max_length=max_length
                    )
                    print(f"{Fore.GREEN}Successfully prepared chat dataset{Style.RESET_ALL}")
                    return tokenized_dataset
                except Exception as e:
                    print(f"{Fore.YELLOW}Warning: Error preparing chat dataset: {str(e)}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Falling back to regular text processing{Style.RESET_ALL}")
                    is_chat_dataset = False  # Fall back to regular processing
            
            if not is_chat_dataset:
                # Handle regular text datasets
                print(f"{Fore.CYAN}Detected regular text dataset{Style.RESET_ALL}")
                
                # Common text field names in datasets
                possible_text_fields = ["text", "content", "sentence", "input", "prompt"]
                possible_text_extends_columns = ['text']
                # Find the first matching text field
                text_field = next((field for field in possible_text_fields if field in available_fields), available_fields[0])
                

                if text_field is None:
                    print(f"{Fore.YELLOW}Available fields in dataset: {available_fields}{Style.RESET_ALL}")
                    raise ValueError("No suitable text field found in dataset. Please check dataset structure.")
                
                print(f"{Fore.CYAN}Using field '{text_field}' for tokenization{Style.RESET_ALL}")
                
                def tokenize_function(examples):
                    # Ensure the input is a string or list of strings
                    texts = examples[text_field]
                    if isinstance(texts, (int, float)):
                        texts = str(texts)
                        
                    elif isinstance(texts, list):
                        holder = []
                        for role,text in zip(examples['role'],texts):
                            combined_text = role + ':' + text
                            holder.append(combined_text)
                        texts = holder
                    elif isinstance(texts, str):
                        #not testing this yet
                        if text_field in possible_text_extends_columns:
                            if 'role' in available_fields:
                                texts = examples['role'] + ':' + texts
                    
                        texts = str(texts)
                    
                    
                    return tokenizer(
                        texts,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    )
                
                tokenized_dataset = dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=dataset["train"].column_names if "train" in dataset else dataset.column_names,
                    num_proc=2
                )
                
                return tokenized_dataset
            
        except Exception as e:
            print(f"{Fore.RED}Error tokenizing dataset: {str(e)}{Style.RESET_ALL}")
            return None
    
    # def model(self):
    #     model = AutoModel.from_pretrained(self.model_id,trust_remote_code=True)
    #     return model
    # def config(self):
    #     config = AutoConfig.from_pretrained(self.model_id,trust_remote_code=True)
    #     return config
    
    def runtuning(self,modelname,datasetname):
        try:
            print(f"{Fore.YELLOW}run tuning:{modelname, datasetname}{Style.RESET_ALL}")
            model, tokenizer = self.load_model(modelname,self.resume_from_checkpoint)
            if model is None or tokenizer is None:
                raise ValueError("Failed to load model or tokenizer")
            
            dataset = self.load_dataset(datasetname)
            tokenized_dataset = self.map_tokenizer(tokenizer, dataset)
            
            trainer = self.Trainer(model=model, dataset=tokenized_dataset, tokenizer=tokenizer,modelname=modelname,datasetname=datasetname)
            trainer.train()
            
            # Save the model in task-specific directory
            model_save_path = self.TASK_MODEL_DIR / modelname.replace('/', '_')
            model_save_path.mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(str(model_save_path))
            tokenizer.save_pretrained(str(model_save_path))
            
            # Save model info for inference
            model_info = {
                "model_id": modelname,
                "model_task": self.model_task,
                "base_model": modelname,
                "finetuned": True,
                "quantization": "4bit",
                "lora_config": {
                    "r": 8,
                    "alpha": 16,
                    "dropout": 0.05
                },
                "last_checkpoint": str(self.last_checkpoint) if self.last_checkpoint else None
            }
            
            with open(model_save_path / "model_info.json", "w") as f:
                json.dump(model_info, f, indent=4)
            
            print(f"{Fore.GREEN}Model saved to: {model_save_path}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error running tuning: {str(e)}{Style.RESET_ALL}")
            report = Report()
            report.store_problem(model=modelname, dataset=datasetname)
            return None
        
    def train_args(self,modelname,datasetname):
        model_folder =  self.CHECKPOINT_DIR / self.model_task
        output_dir = model_folder / modelname if '/' not in modelname else model_folder / modelname.replace('/', '_')
        return TrainingArguments(
            output_dir=output_dir,
            eval_strategy="no",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=0.01,
            max_steps=100,
            save_strategy=self.save_strategy,
            save_total_limit=2,
            save_steps=100,
            save_only_model=True,
            logging_dir=self.CHECKPOINT_DIR,
            logging_strategy="steps",
            logging_steps=100,
            logging_first_step=True,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            remove_unused_columns=False,
            label_names=["labels"],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False,
            ddp_bucket_cap_mb=200,
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
            max_grad_norm=1.0,
            group_by_length=True,
            length_column_name="length",
            report_to="none",
            resume_from_checkpoint=self.last_checkpoint if self.resume_from_checkpoint else None,
            load_best_model_at_end=True,
        )
    def compute_metrics(self,eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)
    
    def Trainer(self, model, dataset, tokenizer,modelname,datasetname):
        # Create data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # We're doing causal language modeling, not masked language modeling
        )
        
        # Handle dataset splitting
        if isinstance(dataset, DatasetDict):
            train_dataset = dataset['train']
        else:
            # If it's a single dataset, use it directly for training
            train_dataset = dataset
        
        return Trainer(
            model=model,
            args=self.train_args(modelname,datasetname),
            train_dataset=train_dataset,
            data_collator=data_collator,
            compute_metrics=None  # Disabled metrics since we're not evaluating
        )

class Manager:
    def __init__(self,model_data_json_path=None):
        self.data_json_path = model_data_json_path
        self.finetune_model = FinetuneModel()
    
    def generate_model_data(self):
        if self.data_json_path is None:
            raise ValueError("data_json_path is required")
        with open(self.data_json_path, "r") as f:
            data = json.load(f)
            return data
    def run_finetune(self,list_model_data):
        try:
            model = None
            dataset = None
            for el in list_model_data:
                # First, load the model
                if "model" in el:
                    model = el["model"]
                    self.finetune_model.load_model(model,self.finetune_model.resume_from_checkpoint)
                
                # Then, combine all datasets for this model
                if "datasets" in el:
                    datasets = el["datasets"]
                    combined_dataset = None
                    for dataset_name in datasets:
                        current_dataset = self.finetune_model.load_dataset(dataset_name)
                        if current_dataset is not None:
                            if combined_dataset is None:
                                combined_dataset = current_dataset
                            else:
                                # Combine datasets if they have the same structure
                                if current_dataset.features == combined_dataset.features:
                                    combined_dataset = concatenate_datasets([combined_dataset, current_dataset])
                    
                    if combined_dataset is not None:
                        
                        self.finetune_model.runtuning(model,combined_dataset)
                
            return model, combined_dataset
        except Exception as e:
            print(f"{Fore.RED}Error running finetune: {str(e)}{Style.RESET_ALL}")
            return model, dataset
                        
                        
# if __name__ == "__main__":
#     HomePath = Path(__file__).parent.parent.absolute()
#     DataModelFolder = f"{HomePath}/DataModel_config"
#     datafile = 'installed.json'
#     model_data_json_path = f"{DataModelFolder}/{datafile}"
#     manager = Manager(model_data_json_path)
#     list_model_data = manager.generate_model_data()
#     manager.run_finetune(list_model_data)
                    
    # finetune_model = FinetuneModel(model_id="")
    # print(finetune_model.tokenizer())
    # print(finetune_model.model())
    # print(finetune_model.config())
    
