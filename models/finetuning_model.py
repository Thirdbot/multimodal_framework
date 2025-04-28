# from langchain.llms import LlamaCpp
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate

from DatasetHandler import DatasetHandler
from trl import SFTTrainer

# from peft.tuners.lora import mark_only_lora_as_trainable

from transformers import AutoTokenizer,BitsAndBytesConfig,AutoConfig,DataCollatorForLanguageModeling,BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments

from peft import LoraConfig, get_peft_model,PeftModel, AutoPeftModelForCausalLM

import torch

from pathlib import Path

import os

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
    def __init__(self, model_id, dataset_name, language=None, split=None):
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.language = language
        self.split = split
        
        # Define paths
        self.WORKSPACE_DIR = Path(__file__).parent.parent.absolute()
        self.MODEL_DIR = self.WORKSPACE_DIR / "models" / "Text-Text-generation"
        self.CHECKPOINT_DIR = self.WORKSPACE_DIR / "checkpoints"
        
        # Create directories
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

        
    def setup_quantization(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_quant_type="nf4"
        )
    
    def setup_lora_config(self):
        return LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]
        )
    
    def setup_training_args(self, dataset):
        """Setup training arguments based on dataset type"""
        # Check if dataset has length
        has_length = hasattr(dataset, '__len__')
        
        # Calculate max_steps if dataset has length
        if has_length:
            dataset_length = len(dataset)
            max_steps = (dataset_length * self.num_train_epochs) // (self.per_device_train_batch_size * self.gradient_accumulation_steps)
        else:
            max_steps = 500  # Default max steps for streaming datasets
            
        return TrainingArguments(
            output_dir=str(self.CHECKPOINT_DIR),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            max_steps=max_steps if not has_length else None,
            gradient_accumulation_steps=4,
            optim='adamw_torch',
            learning_rate=2e-4,
            lr_scheduler_type='cosine',
            warmup_ratio=0.05,
            save_strategy="epoch",
            save_total_limit=3,
            num_train_epochs=3,
            bf16=True,
            save_safetensors=True,
            save_on_each_node=True,
            gradient_checkpointing=True,
            torch_compile=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            label_names=["labels"],
            logging_steps=10,
        )
    
    def load_model_and_tokenizer(self):
        # Load config and tokenizer
        config = AutoConfig.from_pretrained(self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            legacy=False,
            trust_remote_code=True,
            use_fast=True,  # Enable fast tokenizer
            add_prefix_space=True,  # Add space before tokens
            padding_side="right",  # Consistent padding
            truncation_side="right"  # Consistent truncation
        )
        
        # Add special tokens if they don't exist
        special_tokens = {
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>"
        }
        
        # Add any missing special tokens
        for token_name, token_value in special_tokens.items():
            if getattr(tokenizer, token_name) is None:
                tokenizer.add_special_tokens({token_name: token_value})
        
        # Resize token embeddings to match tokenizer
        tokenizer.model_max_length = 2048  # Set maximum sequence length
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=self.setup_quantization(),
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            max_memory={0: "6GB"},
            trust_remote_code=True,
            config=config
        )
        
        # Resize model's token embeddings to match tokenizer
        model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer
    
    def prepare_dataset(self):
        """Load the dataset"""
        dataset = DatasetHandler(self.dataset_name, language=self.language, split=self.split)
        return dataset.download_dataset()
    
    def tokenize_dataset(self, dataset, tokenizer):
        def tokenize(batch):
            return tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def finetune(self):
        # Load model and tokenizer first
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Store tokenizer for later use
        self.tokenizer = tokenizer
        
        # Prepare dataset
        dataset = self.prepare_dataset()
        tokenized_dataset = self.tokenize_dataset(dataset, tokenizer)
        
        # Setup LoRA
        peft_config = self.setup_lora_config()
        model = get_peft_model(model, peft_config)
        
        # Enable gradient checkpointing and input gradients
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        
        # Setup training arguments based on dataset type
        training_args = self.setup_training_args(tokenized_dataset)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            peft_config=peft_config
        )
        
        # Train or load from checkpoint
        if not (os.path.exists(self.CHECKPOINT_DIR) and os.listdir(self.CHECKPOINT_DIR)):
            trainer.train()
            trainer.save_model(self.MODEL_DIR)
            # trainer.model.save_pretrained(self.MODEL_DIR, safe_serialization=True)
            tokenizer.save_pretrained(self.MODEL_DIR)
        else:
            pmodel = PeftModel.from_pretrained(
                model,
                str(self.CHECKPOINT_DIR / "checkpoint-10"),
                is_trainable=True
            )
            merged_model = pmodel.merge_and_unload(safe_merge=True)
            merged_model.save_pretrained(self.MODEL_DIR, safe_serialization=True)
            tokenizer.save_pretrained(self.MODEL_DIR)
        
        return self.MODEL_DIR

# Example usage
if __name__ == "__main__":
    finetuner = FinetuneModel(
        model_id="prometheus-eval/prometheus-7b-v2.0",
        dataset_name="oscar-corpus/OSCAR-2201",
        language='th',
        split='train'
    )
    model_path = finetuner.finetune()
    print(f"Model saved to: {model_path}")