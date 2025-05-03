import os
import json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# from langchain.llms import LlamaCpp
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from trl import SFTTrainer



# from peft.tuners.lora import mark_only_lora_as_trainable

from peft import LoraConfig, get_peft_model,PeftModel, AutoPeftModelForCausalLM

import torch

from pathlib import Path
import evaluate

import numpy as np
from transformers import  AutoModel,AutoTokenizer,AutoConfig,BitsAndBytesConfig,DataCollatorForLanguageModeling,BitsAndBytesConfig, AutoModelForCausalLM, Trainer, TrainingArguments,pipeline

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
        self.per_device_train_batch_size = 1
        self.per_device_eval_batch_size = 1
        self.gradient_accumulation_steps = 4
        self.learning_rate = 2e-4
        self.num_train_epochs = 3
        self.save_strategy = "epoch"
        # Define paths
        self.WORKSPACE_DIR = Path(__file__).parent.parent.absolute()
        self.MODEL_DIR = self.WORKSPACE_DIR / "models" / "Text-Text-generation"
        self.CHECKPOINT_DIR = self.WORKSPACE_DIR / "checkpoints"
        
        # Create directories
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.model_id = None
        self.dataset_name = None
        self.metric = evaluate.load("accuracy")
    
    def load_model(self,model_id):
        print(f"retrieve model {model_id}")
        #load model 
        self.model_id = model_id

    def load_dataset(self,dataset_name):
        print(f"retrieve dataset {dataset_name}")
        #load dataset
        self.dataset_name = dataset_name
        
    def tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return tokenizer
    
    def model(self):
        model = AutoModel.from_pretrained(self.model_id)
        return model
    def config(self):
        config = AutoConfig.from_pretrained(self.model_id)
        return config
    
    def runtuning(self):
        print(f"run tuning:{self.model_id,self.dataset_name}")
        model = self.model()
        tokenizer = self.tokenizer()
        config = self.config()
        dataset = self.load_dataset()
        
    def map_tokenizer(self,tokenizer,dataset,label):
        dataset = dataset.map(lambda x: tokenizer(x[label],padding=True,truncation=True,max_length=512),batched=True)
        return tokenizer
    
    def train_args(self):
        return TrainingArguments(
            output_dir=self.MODEL_DIR,
            evaluation_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=0.01,
            save_strategy=self.save_strategy,
            save_total_limit=2,
            save_steps=100,
            save_only_model=True,
            logging_dir=self.CHECKPOINT_DIR,
            logging_strategy="epoch",
            logging_steps=100,
            logging_first_step=True
        )
    def compute_metrics(self,eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)
    def Trainer(self):
        return Trainer(
            model=self.model(),
            args=self.train_args(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
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
        for el in list_model_data:
            for key,value in el.items():
                if key == "model":
                    model = value
                    self.finetune_model.load_model(model)
                elif key == "datasets":
                    for dataset in value:
                        self.finetune_model.load_dataset(dataset)
                        self.finetune_model.runtuning()
                        
                        
if __name__ == "__main__":
    HomePath = Path(__file__).parent.parent.absolute()
    DataModelFolder = f"{HomePath}/DataModel_config"
    datafile = 'installed.json'
    model_data_json_path = f"{DataModelFolder}/{datafile}"
    manager = Manager(model_data_json_path)
    list_model_data = manager.generate_model_data()
    manager.run_finetune(list_model_data)
                    
    # finetune_model = FinetuneModel(model_id="")
    # print(finetune_model.tokenizer())
    # print(finetune_model.model())
    # print(finetune_model.config())
    
