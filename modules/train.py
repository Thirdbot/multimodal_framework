import os
import json
import torch
from colorama import Fore, Style
from datasets import DatasetDict,load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

from modules.ModelUtils import load_saved_model
from modules.variable import Variable

class FinetuneModel:
    """Class for handling model fine-tuning operations."""
    def __init__(self):
        """Initialize the FinetuneModel with default parameters."""
        # Training parameters
        self.variable = Variable()
        self.per_device_train_batch_size = 1  # Minimal batch size for 6GB GPU
        self.per_device_eval_batch_size = 1
        self.gradient_accumulation_steps = 10  # Accumulate to simulate larger batch
        self.learning_rate = 1e-3
        self.num_train_epochs = 10
        self.save_strategy = "best"
        self.training_config_path = self.variable.training_config_path
        
        # Set memory optimization environment variables
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        
       
        # Initialize paths and directories
        self._setup_directories()
        
        # Initialize components
        self.device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Initialize state variables
        self.model_id = None
        self.dataset_name = None
        self.model_task = None
        

    
    def _setup_directories(self):
        """Set up required directories."""        
        self.variable.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    def train_args(self, task: str, modelname: str) -> TrainingArguments:
        model_folder = self.variable.CHECKPOINT_DIR / task
        
        # Normalize model name
        modelname = modelname.split("\\")[-1] if "custom_models" in modelname else modelname
        output_dir = model_folder / (modelname if '/' not in modelname else modelname.replace('/', '_'))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cuda_available = torch.cuda.is_available()
        
        return TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=0.01,
            save_strategy="steps",
            save_steps=10,
            save_total_limit=1,
            logging_dir=str(output_dir),
            logging_strategy="steps",
            logging_steps=5,
            logging_first_step=True,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=False,  
            bf16=True if cuda_available else False,
            optim="adamw_8bit" if cuda_available else "adamw_torch",  # Use 8-bit optimizer
            lr_scheduler_type="cosine",
            warmup_ratio=0.01,
            remove_unused_columns=False,
            label_names=["labels"],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False,
            ddp_bucket_cap_mb=50,  # Reduced from 200
            dataloader_pin_memory=False,  # Disable to save memory
            dataloader_num_workers=0,
            max_grad_norm=0.5,  # Reduced for stability
            group_by_length=False,  # Disable to save memory
            report_to="none",
            resume_from_checkpoint=True,
            save_safetensors=True,
            save_only_model=True,  # Don't save optimizer state to save memory
            overwrite_output_dir=True,
            torch_compile=False,
            use_mps_device=False,
            eval_strategy="no",
            do_eval=False,
            auto_find_batch_size=False,  # Manual control
            dataloader_prefetch_factor=None,  # Disable prefetching
        )
    
    def Trainer(self, model: AutoModelForCausalLM, dataset: DatasetDict, tokenizer: AutoTokenizer, modelname: str, task: str) -> Trainer:

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
                mlm=False,  # Causal language modeling (not masked)
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
            
            # Ensure model is in training mode with gradients
            model.train()
            # for param in model.parameters():
            #     if hasattr(param, 'requires_grad'):
            #         param.requires_grad = True

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
            
            # Normalize model name for path
            modelname = modelname.replace('/', '_') if '/' in modelname else modelname
            
            # Save model based on type
            if model_type == "vision-model" or "VisionModel" in model_type:
                model_save_path = self.variable.CHECKPOINT_DIR / 'text-vision-text-generation' / modelname
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
                

                
                if hasattr(model, 'vision_adapter'):
                    vision_adapter_path = model_save_path / "vision_adapter"
                    vision_adapter_path.mkdir(parents=True, exist_ok=True)
                    torch.save(model.vision_adapter.state_dict(), str(vision_adapter_path / "vision_adapter.pt"))
                    print(f"{Fore.GREEN}Vision adapter saved to: {vision_adapter_path}{Style.RESET_ALL}")

            elif model_type == "conversation-model" or "ConversationModel" in model_type:
                model_save_path = self.variable.CHECKPOINT_DIR / 'text-generation' / modelname
                model_save_path.mkdir(parents=True, exist_ok=True)
                # Save the final model and adapter
                trainer.save_model(str(model_save_path))
                tokenizer.save_pretrained(str(model_save_path))
                model.config.save_pretrained(str(model_save_path))
            else:
                model_save_path = self.variable.CHECKPOINT_DIR / 'text-generation' / modelname
                model_save_path.mkdir(parents=True, exist_ok=True)
                trainer.save_model(str(model_save_path))
                tokenizer.save_pretrained(str(model_save_path))
                model.config.save_pretrained(str(model_save_path))

            print(f"{Fore.GREEN}Model saved to: {model_save_path}{Style.RESET_ALL}")
            
            # Clean up memory after training
            del trainer
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"{Fore.CYAN}GPU memory cleaned after training{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error running tuning: {str(e)}{Style.RESET_ALL}")
            # Clean up on error too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
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
                dataset = load_from_disk(self.variable.DATASET_FORMATTED_DIR / dataset_format_name)
                # Create model as designed
                if "conversations" in dataset_info:
                    model_name_checkpoint = modelname.replace("/", "_")
                    model_name_path = modelname.split("/")[-1]
                    model_path = self.variable.REGULAR_MODEL_DIR / model_name_path
                    model_task = "text-generation"

                    conversation_checkpoint = self.variable.CHECKPOINT_DIR / model_task / model_name_checkpoint

                    # Load from checkpoint if exists for training only
                    if conversation_checkpoint.exists():
                        print(f"{Fore.GREEN}Loading conversation model from checkpoint...{Style.RESET_ALL}")
                        model, tokenizer = load_saved_model(conversation_checkpoint, checkpoint=True)
                    else:
                        model, tokenizer = load_saved_model(model_path)
                    
                    # Set model to training mode
                    model.train()


                # Handle vision models
                if "image" in dataset_info or "images" in dataset_info:
                    model_name_checkpoint = modelname.replace("/", "_")
                    model_name_path = modelname.split("/")[-1]

                    model_path = self.variable.VISION_MODEL_DIR / model_name_path
                    model_task = "text-vision-text-generation"
                    vision_checkpoint = self.variable.CHECKPOINT_DIR / model_task / model_name_checkpoint

                    if vision_checkpoint.exists():
                        print(f"{Fore.GREEN}Loading vision model from checkpoint...{Style.RESET_ALL}")
                        model, tokenizer = load_saved_model(vision_checkpoint, checkpoint=True)
                    else:
                        model, tokenizer = load_saved_model(model_path)
                    
                    # Ensure vision adapter has gradients enabled
                    if hasattr(model, 'vision_adapter'):
                        for param in model.vision_adapter.parameters():
                            param.requires_grad = True
                    
                    # Set model to training mode with proper gradient setup
                    model.train()
                
                print(f"{Fore.CYAN}Dataset loaded with {len(dataset)} records{Style.RESET_ALL}")

                # Clear CUDA cache before training
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    print(f"{Fore.CYAN}GPU Memory before training: {torch.cuda.memory_allocated()/1e9:.2f} GB{Style.RESET_ALL}")
                
                self.runtuning(model=model, tokenizer=tokenizer, dataset=dataset, modelname=modelname, task=model_task)
