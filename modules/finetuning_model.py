import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import torch
import numpy as np
from colorama import Fore, Style, init
from datasets import load_dataset, concatenate_datasets, DatasetDict
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
from trl import SFTTrainer
import evaluate
from huggingface_hub import HfApi

from modules.defect import Report
from modules.chatTemplate import ChatTemplate
from modules.chainpipe import Chainpipe

# Initialize colorama
init(autoreset=True)

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class FinetuneModel:
    """Class for handling model fine-tuning operations."""
    
    def __init__(self):
        """Initialize the FinetuneModel with default parameters."""
        # Training parameters
        self.per_device_train_batch_size = 64
        self.per_device_eval_batch_size = 64
        self.gradient_accumulation_steps = 2
        self.learning_rate = 2e-4
        self.num_train_epochs = 1
        self.save_strategy = "best"
        
        # Initialize paths and directories
        self._setup_directories()
        
        # Initialize components
        self.device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.metric = evaluate.load("accuracy")
        self.chainpipe = Chainpipe()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Initialize state variables
        self.model_id = None
        self.dataset_name = None
        self.model_task = None
        self.resume_from_checkpoint = True
        self.chat_template = None
        self.last_checkpoint = None
    
    def _setup_directories(self):
        """Set up required directories."""
        self.WORKSPACE_DIR = Path(__file__).parent.parent.absolute()
        self.MODEL_DIR = self.WORKSPACE_DIR / "models"
        self.CHECKPOINT_DIR = self.WORKSPACE_DIR / "checkpoints"
        self.OFFLOAD_DIR = self.WORKSPACE_DIR / "offload"
        
        for directory in [self.MODEL_DIR, self.CHECKPOINT_DIR, self.OFFLOAD_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_model_architecture(self, model_id: str) -> List[str]:
        """Detect the model architecture and return appropriate LoRA configuration.
        
        Args:
            model_id: The model identifier
            
        Returns:
            List of target modules for LoRA configuration
        """
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
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
        
        self.resume_from_checkpoint = resume_from_checkpoint
        print(f"Load from last checkpoint: {self.resume_from_checkpoint}")
        
        try:
            split_name = model_id.split("\\")
            # local_path = Path(model_id)
            if "custom_models" in split_name:
                model_task = split_name[-2]
                self.model_task = model_task
            else:
                self.model_task = self.get_model_task(model_id)
            print(f"{Fore.CYAN}Model task detected: {self.model_task}{Style.RESET_ALL}")
            
            self.TASK_MODEL_DIR = self.MODEL_DIR / self.model_task
            self.TASK_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            
            if resume_from_checkpoint:
                return self._load_from_checkpoint(model_id)
            return self._load_from_scratch(model_id)
            
        except Exception as e:
            print(f"{Fore.RED}Error loading model {model_id}: {str(e)}{Style.RESET_ALL}")
            return None, None
    
    def _load_from_checkpoint(self, model_id: str) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """Load model and tokenizer from checkpoint.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Tuple of (model, tokenizer)
        """
        self.last_checkpoint = self.find_last_checkpoint(model_id)
        if not self.last_checkpoint:
            print(f"{Fore.YELLOW}No valid checkpoint found, starting from scratch{Style.RESET_ALL}")
            return self._load_from_scratch(model_id)
        
        print(f"{Fore.CYAN}Resuming from checkpoint: {self.last_checkpoint}{Style.RESET_ALL}")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            offload_folder=str(self.OFFLOAD_DIR),
            offload_state_dict=True,
            max_memory={0: "40GB"},
            use_cache=False
        )
        
        model = PeftModel.from_pretrained(
            base_model,
            str(self.last_checkpoint),
            device_map=self.device_map,
            torch_dtype=torch.bfloat16
        )
        
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        
        model.gradient_checkpointing_enable()
        
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.last_checkpoint),
            trust_remote_code=True,
            padding_side="right",
            truncation_side="right"
        )
        
        self.chat_template = ChatTemplate(tokenizer=tokenizer)
        
        return model, tokenizer
    
    def _load_from_scratch(self, model_id: str) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """Load model and tokenizer from scratch.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Tuple of (model, tokenizer)
        """     
                
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side="right",
            truncation_side="right"
        )
        
        self.chat_template = ChatTemplate(tokenizer=tokenizer)
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            quantization_config=quantization_config,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            offload_folder=str(self.OFFLOAD_DIR),
            offload_state_dict=True,
            max_memory={0: "40GB"}
        )
        
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        
        target_modules = self.get_model_architecture(model_id)
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        model.train()
        # Only enable gradients for float parameters
        for name, param in model.named_parameters():
            if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                param.requires_grad = True
        
        model.gradient_checkpointing_enable()
        model.print_trainable_parameters()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        
        return model, tokenizer
    
    def load_dataset(self, dataset_name: str, config: Optional[Dict] = None, split: Optional[str] = None) -> Optional[DatasetDict]:
        """Load a dataset.
        
        Args:
            dataset_name: The dataset identifier
            config: Dataset configuration
            split: Dataset split to load
            
        Returns:
            Loaded dataset
        """
        print(f"{Fore.CYAN}Retrieving dataset {dataset_name}{Style.RESET_ALL}")
        
        try:
            self.dataset_name = dataset_name
            if config is not None:
                try:
                    print(f"{Fore.YELLOW}Attempting to load dataset with config: {config}{Style.RESET_ALL}")
                    dataset = load_dataset(dataset_name, config, trust_remote_code=True, split=split)
                    print(f"{Fore.GREEN}Successfully loaded dataset with config{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error loading dataset with config: {str(e)}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Trying to load dataset without config...{Style.RESET_ALL}")
                    try:
                        dataset = load_dataset(dataset_name, trust_remote_code=True, split=split)
                        print(f"{Fore.GREEN}Successfully loaded dataset without config{Style.RESET_ALL}")
                    except Exception as e2:
                        print(f"{Fore.RED}Error loading dataset without config: {str(e2)}{Style.RESET_ALL}")
                        return None
            else:
                try:
                    dataset = load_dataset(dataset_name, trust_remote_code=True, split=split)
                    print(f"{Fore.GREEN}Successfully loaded dataset without config{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error loading dataset: {str(e)}{Style.RESET_ALL}")
                    return None
            return dataset
        except Exception as e:
            print(f"{Fore.RED}Unexpected error loading dataset {dataset_name}: {str(e)}{Style.RESET_ALL}")
            return None
    
    def map_tokenizer(self, dataset_name: str, tokenizer: AutoTokenizer, dataset: DatasetDict, 
                     max_length: int = 384, return_embedded_dataset: bool = False) -> Optional[DatasetDict]:
        """Map tokenizer to dataset.
        
        Args:
            dataset_name: The dataset identifier
            tokenizer: The tokenizer to use
            dataset: The dataset to process
            max_length: Maximum sequence length
            return_embedded_dataset: Whether to return embedded dataset
            
        Returns:
            Processed dataset
        """
        try:
            tokenized_dataset = self.chat_template.prepare_dataset(
                dataset_name,
                dataset,
                max_length=max_length,
                return_embedded_dataset=return_embedded_dataset
            )
            print(f"{Fore.GREEN}Successfully prepared chat dataset{Style.RESET_ALL}")
            return tokenized_dataset
        except Exception as e:
            print(f"{Fore.RED}Error tokenizing dataset: {str(e)}{Style.RESET_ALL}")
            return None
    
    def train_args(self, modelname: str) -> TrainingArguments:
        """Get training arguments.
        
        Args:
            modelname: The model identifier
            
        Returns:
            Training arguments
        """
        model_folder = self.CHECKPOINT_DIR / self.model_task
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
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=0.01,
            save_strategy="steps",
            eval_strategy="steps",
            eval_steps=20,
            save_steps=20,
            save_total_limit=3,
            logging_dir=str(self.CHECKPOINT_DIR),
            logging_strategy="steps",
            logging_steps=20,
            logging_first_step=True,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=False,
            bf16=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            remove_unused_columns=False,
            label_names=["labels"],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False,
            ddp_bucket_cap_mb=200,
            dataloader_pin_memory=cuda_available,
            dataloader_num_workers=2,
            max_grad_norm=1.0,
            group_by_length=True,
            length_column_name="length",
            report_to="none",
            resume_from_checkpoint=self.last_checkpoint if self.resume_from_checkpoint else None,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_safetensors=True,
            save_only_model=True,
            overwrite_output_dir=True,
            torch_compile=False,
            use_mps_device=False
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
    
    def Trainer(self, model: AutoModelForCausalLM, dataset, 
               tokenizer: AutoTokenizer, modelname: str) -> Trainer:
        """Create a trainer instance.
        
        Args:
            model: The model to train
            dataset: The dataset to use
            tokenizer: The tokenizer to use
            modelname: The model identifier
            
        Returns:
            Trainer instance
        """
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        try:
            if isinstance(dataset, DatasetDict):
                train_dataset = dataset.get('train', dataset)
                eval_dataset = dataset.get('test', dataset)
            else:
                split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
                train_dataset = split_dataset['train']
                eval_dataset = split_dataset['test']
        except:
            split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']

        return Trainer(
            model=model,
            args=self.train_args(modelname),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
    
    def runtuning(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                 dataset: DatasetDict, modelname: str) -> None:
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
            trainer = self.Trainer(model=model, dataset=dataset, tokenizer=tokenizer, modelname=modelname)
            trainer.train()
            
            model_save_path = self.TASK_MODEL_DIR / modelname.replace('/', '_')
            model_save_path.mkdir(parents=True, exist_ok=True)
            
            model.save_pretrained(str(model_save_path))
            tokenizer.save_pretrained(str(model_save_path))
            
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
            report.store_problem(model=modelname, dataset=dataset)

class Manager:
    """Manager class for handling fine-tuning operations."""
    
    def __init__(self, model_data_json_path: Optional[str] = None):
        """Initialize the Manager.
        
        Args:
            model_data_json_path: Path to the model data JSON file
        """
        self.data_json_path = model_data_json_path
        self.finetune_model = FinetuneModel()
    
    def generate_model_data(self) -> List[Dict[str, Any]]:
        """Generate model data from JSON file.
        
        Returns:
            List of model data dictionaries
        """
        if self.data_json_path is None:
            raise ValueError("data_json_path is required")
        with open(self.data_json_path, "r") as f:
            return json.load(f)
    
    def run_finetune(self, list_model_data: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[Optional[AutoModelForCausalLM], Optional[DatasetDict]]:
        """Run the fine-tuning process.
        
        Args:
            list_model_data: List of model data dictionaries
            config: Configuration dictionary
            
        Returns:
            Tuple of (model, dataset)
        """
        try:
            model = None
            combined_dataset = None
            dataset = None
            
            for el in list_model_data:
                if "model" in el:
                    modelname = el["model"]
                    model, tokenizer = self.finetune_model.load_model(modelname, self.finetune_model.resume_from_checkpoint)
                
                if "datasets" in el:
                    datasets = el["datasets"]
                    saved_dataset = None
                    first_dataset = None
                    second_dataset = None
                    
                    for dataset_name in datasets:
                        try:
                            print(f"{Fore.CYAN}Loading dataset config: {dataset_name} {config.get(dataset_name, 'No config found')}{Style.RESET_ALL}")
                            
                            dataset = self.finetune_model.load_dataset(dataset_name, config.get(dataset_name, None), split='train')
                           
                            
                            if first_dataset is None:
                                print(f"{Fore.GREEN}Processing first dataset: {dataset_name}{Style.RESET_ALL}")
                                first_dataset = self.finetune_model.map_tokenizer(dataset_name, tokenizer, dataset, return_embedded_dataset=True)
                                concat_dataset = first_dataset
                            else:
                                print(f"{Fore.GREEN}Processing additional dataset: {dataset_name}{Style.RESET_ALL}")
                                second_dataset = self.finetune_model.map_tokenizer(dataset_name, tokenizer, dataset, return_embedded_dataset=True)
                                if first_dataset is not None and second_dataset is not None:
                                    print(f"{Fore.GREEN}Concatenating datasets...{Style.RESET_ALL}")
                                    # concat_dataset = concatenate_datasets([first_dataset, second_dataset])
                                    # Get column names from both datasets
                                    first_cols = set(first_dataset.column_names)
                                    second_cols = set(second_dataset.column_names)
                                    print(f"{Fore.GREEN}First dataset columns: {first_cols}{Style.RESET_ALL}")
                                    print(f"{Fore.GREEN}Second dataset columns: {second_cols}{Style.RESET_ALL}")
                                    
                                    # For columns only in second dataset, add them to first dataset with None values
                                    for col in second_cols - first_cols:
                                        first_dataset = first_dataset.add_column(col, [None] * len(first_dataset))
                                    
                                    # For columns only in first dataset, add them to second dataset with None values  
                                    for col in first_cols - second_cols:
                                        second_dataset = second_dataset.add_column(col, [None] * len(second_dataset))
                                    
                                    # Now both datasets have same columns, concatenate them
                                    concat_dataset = concatenate_datasets([first_dataset, second_dataset])
                                    
                                    print(f"{Fore.GREEN}Successfully joined datasets with columns: {concat_dataset.column_names}{Style.RESET_ALL}")
                                    
                            
                            saved_dataset = self.finetune_model.map_tokenizer(dataset_name, tokenizer, concat_dataset, return_embedded_dataset=False)
                            
                        except Exception as e:
                            print(f"{Fore.RED}Error processing dataset {dataset_name}: {str(e)}{Style.RESET_ALL}")
                            print(f"{Fore.YELLOW}Stack trace:", exc_info=True)
                            continue
                    
                    dataset = saved_dataset
                
                if model is not None and dataset is not None:
                    self.finetune_model.runtuning(model, tokenizer, dataset, modelname)
            
            return model, dataset
            
        except Exception as e:
            print(f"{Fore.RED}Error running finetune: {str(e)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Stack trace:", exc_info=True)
            return None, None
