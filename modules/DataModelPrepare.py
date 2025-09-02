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

class FinetuneModel:
    """Class for handling model fine-tuning operations."""
    
    def __init__(self):
        """Initialize the FinetuneModel with default parameters."""
        # Training parameters
        self.variable = Variable()
        self.per_device_train_batch_size = 1  # Reduced batch size
        self.per_device_eval_batch_size = 1
        self.gradient_accumulation_steps = 2  # Reduced gradient accumulation
        self.learning_rate = 2e-5  # Reduced learning rate
        self.num_train_epochs = 100
        self.save_strategy = "best"
        
       
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

        self.chat_template = None

        self.CUTOM_MODEL_DIR = self.variable.CUTOM_MODEL_DIR
        self.VISION_MODEL_DIR = self.variable.VISION_MODEL_DIR
        self.REGULAR_MODEL_DIR = self.variable.REGULAR_MODEL_DIR
        self.MODEL_LOCAL_DIR = self.variable.REPO_DIR
        
        
        

    
    def _setup_directories(self):
        """Set up required directories."""        
        self.CHECKPOINT_DIR = self.variable.CHECKPOINT_DIR
        self.OFFLOAD_DIR = self.variable.OFFLOAD_DIR
        
        for directory in [self.CHECKPOINT_DIR, self.OFFLOAD_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_model_architecture(self, model_id: str) -> List[str]:
        """Detect the model architecture and return appropriate LoRA configuration.
        
        Args:
            model_id: The model identifier
            
        Returns:
            List of target modules for LoRA configuration
        """
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
        """Get the task type for a model.
        
        Args:
            model_name: The model identifier
            
        Returns:
            The model's task type
        """
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
    

    def find_last_checkpoint(self, model_name: str) -> Optional[Path]:
        """Find the last checkpoint for a model.
        
        Args:
            model_name: The model identifier
            
        Returns:
            Path to the last checkpoint if found, None otherwise
        """
        try:
            model_task = self.get_model_task(model_name)
            model_name = model_name.replace('/', '_') if '/' in model_name else model_name
            
            checkpoint_dir = self.CHECKPOINT_DIR / model_task / model_name
            
            if not checkpoint_dir.exists():
                print(f"{Fore.YELLOW}No checkpoint directory found at {checkpoint_dir}{Style.RESET_ALL}")
                return None
            
            checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
            
            if not checkpoints:
                print(f"{Fore.YELLOW}No checkpoints found in {checkpoint_dir}{Style.RESET_ALL}")
                return None
            
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
            
            if not (latest_checkpoint / "model.safetensors").exists() and not (latest_checkpoint / "adapter_model.safetensors").exists():
                print(f"{Fore.YELLOW}Latest checkpoint {latest_checkpoint} is incomplete, starting from scratch{Style.RESET_ALL}")
                return None
                
            print(f"{Fore.GREEN}Found valid checkpoint at {latest_checkpoint}{Style.RESET_ALL}")
            return latest_checkpoint
            
        except Exception as e:
            print(f"{Fore.RED}Error finding last checkpoint: {str(e)}{Style.RESET_ALL}")
            return None
    
    
    def load_model(self, model_id: str, resume_from_checkpoint: bool = False) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """Load a model and its tokenizer.
        
        Args:
            model_id: The model identifier
            resume_from_checkpoint: Whether to resume from a checkpoint
            
        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"{Fore.CYAN}Retrieving model {model_id}{Style.RESET_ALL}")
        

        # self.resume_from_checkpoint = resume_from_checkpoint
        # print(f"Load from last checkpoint: {self.resume_from_checkpoint}")

        
        try:
            # Check if model_id is a local path or Hugging Face model ID
            # if Path(model_id).exists():
            #     split_name = model_id.split("\\")
            #     if "custom_models" in split_name:
            #         model_task = split_name[-2]
            #         self.model_task = model_task
            # else:
            self.model_task = self.get_model_task(model_id)
            print(f"{Fore.CYAN}Model task detected: {self.model_task}{Style.RESET_ALL}")
            
            # self.TASK_MODEL_DIR = self.MODEL_DIR.joinpath(self.model_task)
            # self.TASK_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            

            # if resume_from_checkpoint:
            #     return self._load_from_checkpoint(model_id)

            return self._load_from_scratch(model_id)


        except Exception as e:
            print(f"{Fore.RED}Error loading model {model_id}: {str(e)}{Style.RESET_ALL}")
            return None, None
    

    # def _load_from_checkpoint(self, model_id: str) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    #     """Load model and tokenizer from checkpoint.
        
    #     Args:
    #         model_id: The model identifier
            
    #     Returns:
    #         Tuple of (model, tokenizer)
    #     """
    #     # self.last_checkpoint = self.find_last_checkpoint(model_id)
    #     if not self.last_checkpoint:
    #         print(f"{Fore.YELLOW}No valid checkpoint found, starting from scratch{Style.RESET_ALL}")
    #         return self._load_from_scratch(model_id)
        
    #     print(f"{Fore.CYAN}Resuming from checkpoint: {self.last_checkpoint}{Style.RESET_ALL}")
        
    #     try:
    #         # Configure quantization
    #         bnb_config = BitsAndBytesConfig(
    #             load_in_4bit=True,
    #             bnb_4bit_compute_dtype=torch.float32,
    #             bnb_4bit_quant_type="fp4",
    #             bnb_4bit_use_double_quant=False
    #         )
            
    #         # Load base model with quantization
    #         base_model = AutoModelForCausalLM.from_pretrained(
    #             model_id,
    #             device_map=self.device_map,
    #             trust_remote_code=True,
    #             quantization_config=bnb_config,
    #             torch_dtype=torch.float32
    #         )
            
    #         # Prepare model for k-bit training
    #         base_model = prepare_model_for_kbit_training(base_model)
            
    #         # Load tokenizer from checkpoint
    #         tokenizer = AutoTokenizer.from_pretrained(
    #             str(self.last_checkpoint),
    #             trust_remote_code=True,
    #             padding_side="right",
    #             truncation_side="right"
    #         )
            
    #         # Load the PEFT model configuration and weights
    #         model = AutoPeftModelForCausalLM.from_pretrained(
    #             str(self.last_checkpoint),
    #             device_map=self.device_map,
    #             torch_dtype=torch.float32,
    #             is_trainable=True
    #         )
            
    #         model.train()
    #         model.gradient_checkpointing_enable()
            
    #         self.chat_template = ChatTemplate(tokenizer=tokenizer)
            
    #         print(f"{Fore.GREEN}Successfully loaded checkpoint with LoRA configuration{Style.RESET_ALL}")
    #         return model, tokenizer
            
    #     except Exception as e:
    #         print(f"{Fore.RED}Error loading from checkpoint: {str(e)}{Style.RESET_ALL}")
    #         print(f"{Fore.YELLOW}Attempting to load from scratch...{Style.RESET_ALL}")
    #         return self._load_from_scratch(model_id)
    
    def _load_from_scratch(self, model_id: str) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """Load model and tokenizer from scratch.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Tuple of (model, tokenizer)
        """ 
        # model_name = model_id.replace('/', '--') if '/' in model_id else model_id
        
        # model_path = self.variable.LocalModel_DIR / ('models--'+ model_name)
        model_path = self.variable.LocalModel_DIR / model_id
        try:
            print(f"{Fore.CYAN}Downloading and loading model: {model_path}{Style.RESET_ALL}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="right",
                truncation_side="right",
                
            )
            
            self.chat_template = ChatTemplate(tokenizer=tokenizer)
            
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
                r=32,  # Rank
                lora_alpha=64,  # Alpha scaling
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
        """Load a dataset.
        
        Args:
            dataset_name: The dataset identifier
            config: Dataset configuration
            split: Dataset split to load
            
        Returns:
            Loaded dataset
        """
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
    
    def map_tokenizer(self, dataset_name: str, tokenizer: AutoTokenizer, dataset: DatasetDict, 
                     max_length: int = 384, Tokenizing: bool = False) -> Optional[DatasetDict]:

        print(f"{Fore.CYAN}Processing dataset with max length: {max_length}{Style.RESET_ALL}")
        
        # Ensure tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"{Fore.GREEN}Set padding token to EOS token{Style.RESET_ALL}")
            
        self.chat_template = ChatTemplate(tokenizer=tokenizer)
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
    

    def train_args(self,task:str, modelname: str) -> TrainingArguments:

        """Get training arguments.
        
        Args:
            modelname: The model identifier
            
        Returns:
            Training arguments
        """

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
            save_steps=100,
            save_total_limit=2,
            logging_dir=str(self.CHECKPOINT_DIR),
            logging_strategy="steps",
            logging_steps=10,
            logging_first_step=True,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=False,  # Disable fp16 since we're using bf16
            bf16=True,  # Use bfloat16 instead of fp16
            optim="adamw_torch_fused" if cuda_available else "adamw_torch",
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
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

            resume_from_checkpoint=self.CHECKPOINT_DIR,

            save_safetensors=True,
            save_only_model=False,  # Changed to False to save optimizer state
            overwrite_output_dir=True,
            torch_compile=False,
            use_mps_device=False,
            eval_strategy="no",  # Disable evaluation completely
            do_eval=False  # Ensure evaluation is disabled
        )
    
    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dictionary of metrics
        """
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
    

    def Trainer(self, model: AutoModelForCausalLM, dataset, tokenizer: AutoTokenizer, modelname: str,task:str) -> Trainer:

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
                mlm=False,  # We want causal language modeling, not masked
                pad_to_multiple_of=8  # For better GPU utilization
            )
            
            print(f"{Fore.CYAN}Setting up training dataset{Style.RESET_ALL}")
            
            if isinstance(dataset, DatasetDict) and 'train' in dataset:
                print(f"{Fore.GREEN}Using provided train split{Style.RESET_ALL}")
                train_dataset = dataset['train']
                print(f"Train size: {len(train_dataset)}")
            else:
                print(f"{Fore.YELLOW}Using entire dataset for training{Style.RESET_ALL}")
                train_dataset = dataset
                print(f"Train size: {len(train_dataset)}")

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

        """Run the fine-tuning process.
        
        Args:
            model: The model to train
            tokenizer: The tokenizer to use
            dataset: The dataset to use
            modelname: The model identifier
        """
        try:
            if "custom_models" in modelname.split("/"):
                modelname = modelname.split("/")
                modelname = modelname[-1]

            trainer = self.Trainer(model=model, dataset=dataset, tokenizer=tokenizer, modelname=modelname,task=task)
            
            # Save the initial LoRA config
            model.save_pretrained(trainer.args.output_dir)
            print("Saving model...")
            
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
                model.save_pretrained(str(model_save_path), safe_serialization=True)
                tokenizer.save_pretrained(str(model_save_path))
                if hasattr(model, "lang_model"):
                    lang_model_path = model_save_path / "lang_model"
                    lang_model_path.mkdir(parents=True, exist_ok=True)
                    model.lang_model.save_pretrained(str(lang_model_path))
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
                model.save_pretrained(str(model_save_path), safe_serialization=True)
                tokenizer.save_pretrained(str(model_save_path))
                model.config.save_pretrained(str(model_save_path))
            else:
                model_save_path = self.CHECKPOINT_DIR / 'text-generation' / modelname
                model_save_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(model_save_path), safe_serialization=True)
                tokenizer.save_pretrained(str(model_save_path))
                model.config.save_pretrained(str(model_save_path))


            # # Get target modules and ensure they're serializable
            # target_modules = model.peft_config["default"].target_modules
            # if isinstance(target_modules, (set, list, tuple)):
            #     target_modules = list(target_modules)
            # else:
            #     target_modules = []
          

            print(f"{Fore.GREEN}Model saved to: {model_save_path}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error running tuning: {str(e)}{Style.RESET_ALL}")
            

class Manager:
    """Manager class for handling fine-tuning operations."""
    
    def __init__(self):
        """Initialize the Manager.
        
        Args:
            model_data_json_path: Path to the model data JSON file
        """
        self.variable = Variable()
        self.finetune_model = FinetuneModel()
        
        self.repository = self.variable.REPO_DIR

    
    def dataset_prepare(self, list_model_data: List[Dict[str, Any]]) -> Tuple[Optional[AutoModelForCausalLM], Optional[DatasetDict]]:
        """Run the fine-tuning process.
        
        Args:
            list_model_data: List of model data dictionaries
            config: Configuration dictionary
            
        Returns:
            Tuple of (model, dataset)
        """
        
        datamodel_file = self.variable.SAVED_CONFIG_Path
        
        datamodel_file = datamodel_file.as_posix()
        
        
        try:
            with open(datamodel_file, 'r') as f:
                config = json.load(f)
        except:
            print(f"error config file not found {datamodel_file}")
            
        try:
            model = None
            # combined_dataset = None
            dataset = None
            saved_dataset = None
            print(list_model_data)
            
            #load model and dataset prepare for tuning
            for modelname,dict_dataset in list_model_data['model'].items():
                    
                    #load tokenizer

                    model, tokenizer = self.finetune_model.load_model(modelname)

                
                    saved_dataset = None
                    first_dataset = None
                    second_dataset = None
                    
                    first_cols = set()
                    second_cols = set()
                    concat_dataset = set()
                    
                    for dataset_name,info in dict_dataset.items():
                        try:
                            print(f"{Fore.CYAN}Loading dataset config: {dataset_name} {config.get(dataset_name, 'No config found')}{Style.RESET_ALL}")
                            
                            dataset = self.finetune_model.load_dataset(dataset_name, config.get(dataset_name, 'default'))
                           
                            
                            if first_dataset is None:
                                print(f"{Fore.GREEN}Processing first dataset: {dataset_name}{Style.RESET_ALL}")
                                
                                #return processed True make it return text
                                first_dataset = self.finetune_model.map_tokenizer(dataset_name, 
                                                                               tokenizer, dataset, 
                                                                               Tokenizing=False)
                                if first_dataset is None:
                                    print(f"{Fore.RED}Failed to process first dataset: {dataset_name}{Style.RESET_ALL}")
                                    continue
                                first_cols = set(first_dataset.column_names)
                                concat_dataset = first_dataset
                                
                                
                                
                                            
                            else:
                                first_dataset = concat_dataset
                                print(f"{Fore.GREEN}Processing additional dataset: {dataset_name}{Style.RESET_ALL}")
                                
                                #return processed True make it return text
                                second_dataset = self.finetune_model.map_tokenizer(dataset_name, 
                                                                                tokenizer, 
                                                                                dataset, 
                                                                                Tokenizing=False)

                                second_dataset = dataset
                                if second_dataset is None:
                                    print(f"{Fore.RED}Failed to process second dataset: {dataset_name}{Style.RESET_ALL}")
                                    continue

                                second_cols = set(second_dataset.column_names)
                                
                                
                                print(f"{Fore.GREEN}Concatenating datasets...{Style.RESET_ALL}")
                             
                                print(f"{Fore.GREEN}First dataset columns: {first_cols}{Style.RESET_ALL}")
                                print(f"{Fore.GREEN}Second dataset columns: {second_cols}{Style.RESET_ALL}")
                                
                            
                                        
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
                                

                            
                            
                        except Exception as e:
                            print(f"{Fore.RED}Error processing dataset {dataset_name}: {str(e)}{Style.RESET_ALL}")
                            continue
                    
                    dataset = concat_dataset
                    
                    pd.DataFrame(dataset).to_csv(Path(__file__).parent.parent.absolute() / "concat_dataset.csv")
                    
                    union_cols = first_cols.union(second_cols)


                    #after formatted to right format it use to embedding
                    #after getting concatenate dataset return it to embedding formatted with return both false since the model going to tokenized it anyways
                    saved_dataset = self.finetune_model.map_tokenizer(dataset_name, 
                                                                    tokenizer, 
                                                                    dataset,
                                                                    Tokenizing=True)

                    pd.DataFrame(saved_dataset).to_csv(Path(__file__).parent.parent.absolute() / "embedded_dataset.csv")
                    
                    # model_local_path = self.repository / modelname
                    
                    #create model as design
                    if "conversations" in union_cols:
                        #if model is not local and been createdd
                        model_name_safe = modelname.replace("/","_")
                        model_path = self.finetune_model.REGULAR_MODEL_DIR / model_name_safe
                        model_task = "text-generation"

                        self.CHECKPOINT_DIR = self.finetune_model.CHECKPOINT_DIR / model_task / model_name_safe
                        #if it local created model

                        if not (model_path).exists():
                            print(f"{Fore.GREEN}Creating conversation model...from {modelname}{Style.RESET_ALL}")
                            create_model = CreateModel(modelname, "conversation-model")
                            create_model.add_conversation()
                            create_model.save_regular_model()
                            # model, tokenizer = load_saved_model(model_path)
      

                        elif Path(self.CHECKPOINT_DIR).exists():
                            print(f"{Fore.GREEN}Loading conversation model from checkpoint...{Style.RESET_ALL}")
                            model, tokenizer = load_saved_model(self.CHECKPOINT_DIR)


                    #temporal fix this
                    if "image" in union_cols or "images" in union_cols:
                        model_name_safe = modelname.replace("/","_")

                        model_path = self.finetune_model.VISION_MODEL_DIR / model_name_safe                       
                        model_task = "text-vision-text-generation"
                        self.CHECKPOINT_DIR = self.finetune_model.CHECKPOINT_DIR / model_task / model_name_safe

                        if not (model_path).exists():
                            print(f"{Fore.GREEN}Creating vision model...from {modelname}{Style.RESET_ALL}")
                            create_model = CreateModel(modelname, "vision-model")
                            create_model.add_vision()
                            create_model.save_vision_model()
                            # model, tokenizer = load_saved_model(model_path)
                            


                        elif Path(self.CHECKPOINT_DIR).exists():
                            print(f"{Fore.GREEN}Loading vision model from checkpoint...{Style.RESET_ALL}")
                            model, tokenizer = load_saved_model(self.CHECKPOINT_DIR)
                    
                    model, tokenizer = load_saved_model(model_path)

                 
                    ## run finetuning part
                    if model is not None and saved_dataset is not None:
                        self.finetune_model.runtuning(model, tokenizer, saved_dataset, modelname, model_task)

            return model, saved_dataset
            
        except Exception as e:
            print(f"{Fore.RED}Error running finetune: {str(e)}{Style.RESET_ALL}")
            return None, None
