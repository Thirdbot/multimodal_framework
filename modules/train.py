import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import torch
import numpy as np
from colorama import Fore, Style, init
from datasets import load_dataset, concatenate_datasets, DatasetDict,get_dataset_config_info,get_dataset_split_names,get_dataset_config_names,load_from_disk
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
from modules.createbasemodel import load_saved_model

from modules.variable import Variable
import ast
import pandas as pd

class FinetuneModel:
    """Class for handling model fine-tuning operations."""
    
    def __init__(self):
        """Initialize the FinetuneModel with default parameters."""
        # Training parameters
        self.variable = Variable()
        self.per_device_train_batch_size = 1  # Reduced batch size
        self.per_device_eval_batch_size = 1
        self.gradient_accumulation_steps = 1  # Reduced gradient accumulation
        self.learning_rate = 2e-4
        self.num_train_epochs = 6
        self.save_strategy = "best"
        self.training_config_path = self.variable.training_config_path
        
       
        # Initialize paths and directories
        self._setup_directories()
        
        # Initialize components
        self.device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.metric = evaluate.load("accuracy")
        # self.chainpipe = Chainpipe()
        
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Initialize state variables
        self.model_id = None
        self.dataset_name = None
        self.model_task = None


        self.CUTOM_MODEL_DIR = self.variable.CUTOM_MODEL_DIR
        self.VISION_MODEL_DIR = self.variable.VISION_MODEL_DIR
        self.REGULAR_MODEL_DIR = self.variable.REGULAR_MODEL_DIR
        self.MODEL_LOCAL_DIR = self.variable.REPO_DIR
        
        self.dataset_formatted_dir = self.variable.DATASET_FORMATTED_DIR
        

    
    def _setup_directories(self):
        """Set up required directories."""        
        self.CHECKPOINT_DIR = self.variable.CHECKPOINT_DIR
        
        for directory in [self.CHECKPOINT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def train_args(self,task:str, modelname: str) -> TrainingArguments:

        model_folder = self.CHECKPOINT_DIR / task

        if "custom_models" in modelname.split("\\"):
            modelname = modelname.split("\\")
            modelname = modelname[-1]
        output_dir = model_folder / modelname if '/' not in modelname else model_folder / modelname.replace('/', '_')
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cuda_available = torch.cuda.is_available()
        
        return TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=0.01,
            save_strategy="steps",
            save_steps=5,
            save_total_limit=1,
            logging_dir=str(output_dir),
            logging_strategy="steps",
            logging_steps=5,
            logging_first_step=True,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=False,  
            bf16=True,  
            optim="adamw_torch_fused" if cuda_available else "adamw_torch",
            lr_scheduler_type="cosine",
            warmup_ratio=0.01,
            remove_unused_columns=False,
            label_names=["labels"],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False,
            ddp_bucket_cap_mb=200,
            dataloader_pin_memory=cuda_available,
            dataloader_num_workers=0,
            max_grad_norm=1.0,
            group_by_length=True,
            length_column_name="length",
            report_to="none",
            resume_from_checkpoint=True,
            save_safetensors=True,
            save_only_model=False,  # Changed to False to save optimizer state
            overwrite_output_dir=True,
            torch_compile=False,
            use_mps_device=False,
            eval_strategy="no",  # Disable evaluation completely
            do_eval=False  # Ensure evaluation is disabled
        )
    
    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
     
        logits, labels = eval_pred
        predictions = np.argmax(logits[:, -1, :], axis=-1)
        valid_labels = labels[:, -1]
        mask = valid_labels != -100
        filtered_predictions = predictions[mask]
        filtered_labels = valid_labels[mask]
        
        if len(filtered_predictions) == 0 or len(filtered_labels) == 0:
            return {"accuracy": 0.0}
            
        try:
            metrics = self.metric.compute(predictions=filtered_predictions, references=filtered_labels)
            if metrics is None or np.isnan(metrics.get("accuracy", 0.0)):
                return {"accuracy": 0.0}
            return metrics
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Error computing metrics: {str(e)}{Style.RESET_ALL}")
            return {"accuracy": 0.0}
    
    def parse_dict(self,example):
        return ast.literal_eval(example['train'])



    def Trainer(self, model: AutoModelForCausalLM, dataset:DatasetDict, tokenizer: AutoTokenizer, modelname: str,task:str) -> Trainer:

        try:
            """Create a trainer instance."""
            # Print model parameters
            trainable_params = 0
            all_param = 0
            for _, param in model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            print(f"{Fore.CYAN}Trainable params: {trainable_params:,} ({100 * trainable_params / all_param:.2f}%){Style.RESET_ALL}")
            print(f"{Fore.CYAN}All params: {all_param:,}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Non-trainable params: {all_param - trainable_params:,}{Style.RESET_ALL}")
            
            # Configure data collator for language modeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # We want masked language modeling
                pad_to_multiple_of=8  # For better GPU utilization
            )
            
            print(f"{Fore.CYAN}Setting up training dataset{Style.RESET_ALL}")
            train_dataset = dataset['train']
            print(f"{Fore.CYAN}Training dataset size: {len(train_dataset)}{Style.RESET_ALL}")
            
            return Trainer(
                model=model,

                args=self.train_args(task,modelname),

                train_dataset=train_dataset,
                data_collator=data_collator,
            )
        except Exception as e:
            print(f"{Fore.RED}Error creating trainer: {str(e)}{Style.RESET_ALL}")
            return None
        
    def runtuning(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 

                 dataset: DatasetDict, modelname: str,task:str) -> None:
        try:
            if "custom_models" in modelname.split("/"):
                modelname = modelname.split("/")
                modelname = modelname[-1]

            trainer = self.Trainer(model=model, dataset=dataset, tokenizer=tokenizer, modelname=modelname,task=task)
            
            # Save the initial LoRA config
            # model.save_pretrained(trainer.args.output_dir)
            # print("Saving model...")
            
            # Start training
            trainer.train()

            if hasattr(model, "config"):
                if hasattr(model.config, "model_type"):
                    model_type = model.config.model_type
                else:
                    model_type = "conversation-model"
                    model.config.model_type = "conversation-model"
            else:
                model_type = "conversation-model"
                model.config.model_type = "conversation-model"

            print(f"{Fore.CYAN}Identified model type to save: {model_type}{Style.RESET_ALL}")
            
            
            modelname = modelname.replace('/', '_') if '/' in modelname else modelname
            
            
            #save model needed outside checkpoints
            if model_type == "vision-model" or "VisionModel" in model_type:
                model_save_path = self.CHECKPOINT_DIR / 'text-vision-text-generation' / modelname
                model_save_path.mkdir(parents=True, exist_ok=True)
                # Save the final model and adapter
                trainer.save_model(str(model_save_path))
                tokenizer.save_pretrained(str(model_save_path))
                if hasattr(model, "lang_model"):
                    lang_model_path = model_save_path / "lang_model"
                    lang_model_path.mkdir(parents=True, exist_ok=True)
                    
                    model.lang_model.save_pretrained(str(lang_model_path))
                    tokenizer.save_pretrained(str(lang_model_path))
                    print(f"{Fore.GREEN}Language model saved to: {lang_model_path}{Style.RESET_ALL}")
                
                if hasattr(model, "vision_model"):
                    vision_model_path = model_save_path / "vision_model"
                    vision_model_path.mkdir(parents=True, exist_ok=True)
                    model.vision_model.save_pretrained(str(vision_model_path))
                    print(f"{Fore.GREEN}Vision model saved to: {vision_model_path}{Style.RESET_ALL}")
                
                if hasattr(model, "vision_processor"):
                    vision_processor_path = model_save_path / "vision_processor"
                    vision_processor_path.mkdir(parents=True, exist_ok=True)
                    model.vision_processor.save_pretrained(str(vision_processor_path))
                    print(f"{Fore.GREEN}Vision processor saved to: {vision_processor_path}{Style.RESET_ALL}")
                
            elif model_type == "conversation-model" or "ConversationModel" in model_type:
                model_save_path = self.CHECKPOINT_DIR / 'text-generation' / modelname
                model_save_path.mkdir(parents=True, exist_ok=True)
                # Save the final model and adapter
                trainer.save_model(str(model_save_path))
                tokenizer.save_pretrained(str(model_save_path))
                model.config.save_pretrained(str(model_save_path))
            else:
                model_save_path = self.CHECKPOINT_DIR / 'text-generation' / modelname
                model_save_path.mkdir(parents=True, exist_ok=True)
                trainer.save_model(str(model_save_path))
                tokenizer.save_pretrained(str(model_save_path))
                model.config.save_pretrained(str(model_save_path))

            print(f"{Fore.GREEN}Model saved to: {model_save_path}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error running tuning: {str(e)}{Style.RESET_ALL}")
    
    def load_for_tuning(self):
        try:
            with open(self.training_config_path, 'r') as f:
                model_training_data = json.load(f)
            return model_training_data
        except Exception as e:
            print(f"{Fore.RED}Error loading training config: {str(e)}{Style.RESET_ALL}")
            return None
    
    def finetune_model(self):
        model_training_data = self.load_for_tuning()

        for modelname, dict_dataset in model_training_data['model'].items():
            print(f"{Fore.CYAN}Preparing to fine-tune model: {modelname}{Style.RESET_ALL}")

            for dataset_name, dataset_info in dict_dataset.items():
                dataset_format_name = f"{dataset_name.replace('/', '_')}_formatted"
                

                dataset = load_from_disk(self.dataset_formatted_dir / dataset_format_name)
                #create model as design
                if "conversations" in dataset_info:
                    #if model is not local and been createdd
                    model_name_safe = modelname.replace("/","_")
                    model_path = self.REGULAR_MODEL_DIR / model_name_safe
                    model_task = "text-generation"

                    self.conversation_checkpoint = self.CHECKPOINT_DIR / model_task / model_name_safe

                    # load from checkpoint if exists for training only
                    if Path(self.conversation_checkpoint).exists():
                        print(f"{Fore.GREEN}Loading conversation model from checkpoint...{Style.RESET_ALL}")
                        model, tokenizer = load_saved_model(self.conversation_checkpoint,checkpoint=True)
                    else:
                        model, tokenizer = load_saved_model(model_path)


                #temporal fix this
                if "image" in dataset_info or "images" in dataset_info:
                    model_name_safe = modelname.replace("/","_")

                    model_path = self.VISION_MODEL_DIR / model_name_safe                       
                    model_task = "text-vision-text-generation"
                    self.vision_checkpoint = self.CHECKPOINT_DIR / model_task / model_name_safe
                        

                    if Path(self.vision_checkpoint).exists():
                        print(f"{Fore.GREEN}Loading vision model from checkpoint...{Style.RESET_ALL}")
                        model, tokenizer = load_saved_model(self.vision_checkpoint,checkpoint=True)
                    else:
                        model, tokenizer = load_saved_model(model_path)
                print(f"{Fore.CYAN}Dataset loaded with {len(dataset)} records{Style.RESET_ALL}")
                self.runtuning(model=model, tokenizer=tokenizer, dataset=dataset, modelname=modelname,task=model_task)
