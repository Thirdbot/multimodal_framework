import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# from langchain.llms import LlamaCpp
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from trl import SFTTrainer
try:
    from models.DatasetHandler  import LangDataset
    
except:
    from DatasetHandler  import LangDataset

# from peft.tuners.lora import mark_only_lora_as_trainable

from transformers import AutoTokenizer,BitsAndBytesConfig,AutoConfig,DataCollatorForLanguageModeling,BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments,pipeline

from peft import LoraConfig, get_peft_model,PeftModel, AutoPeftModelForCausalLM

import torch

from pathlib import Path

from transformers import  LlamaForCausalLM

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
    def __init__(self, model_id=None, dataset_name=None, language=None, split=None):
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.language = language
        self.split = split
        
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
   