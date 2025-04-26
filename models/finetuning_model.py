# from langchain.llms import LlamaCpp
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate

from DatasetHandler import DatasetHandler

from trl import SFTTrainer

# from peft.tuners.lora import mark_only_lora_as_trainable

from transformers import AutoTokenizer,DataCollatorForLanguageModeling,BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments

from peft import LoraConfig, get_peft_model,PeftModel

import torch

from pathlib import Path

import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define absolute paths
WORKSPACE_DIR = Path(__file__).parent.parent.absolute()
MODEL_DIR = WORKSPACE_DIR / "models" / "Text-Text-generation"
CHECKPOINT_DIR = WORKSPACE_DIR / "checkpoints"

# Create model directory if it doesn't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)


model_id = "Orenguteng/Llama-3-8B-Lexi-Uncensored"


#multilingual dataset
dataset = DatasetHandler("oscar-corpus/OSCAR-2201",language='th',split='train')
#get dataset (im prove later)
ds = dataset.get_dataset()


#quantiation for load 4 bit
q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True,
    bnb_4bit_quant_type="nf4"  # Using nf4 for better memory efficiency
)


#model finetuning 
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    legacy=False,
)

# Load base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=q_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    max_memory={0: "6GB"},
    use_cache=False  # Disable cache for gradient checkpointing compatibility
)

# Configure LoRA
peft_config = LoraConfig(
    r=4,  # Reduced from 8 for faster training
    lora_alpha=8,  # Reduced proportionally with r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],  # Focus on key attention layers
    modules_to_save=None
)

# Get PEFT model - only apply PEFT once
if not hasattr(model, 'peft_config'):
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

#map token
tokenized_ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

#training args specifiction using bfp16 for precision 2x save .pth file
training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    max_steps=10,
    gradient_accumulation_steps=4,
    optim='adamw_torch',
    learning_rate=2e-4,  # Slightly higher learning rate
    lr_scheduler_type='cosine',  # Better learning rate schedule
    warmup_ratio=0.05,  # Reduced warmup
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
    logging_steps=10,            # Log every 10 steps
)
#collator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

# Configure trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=collator,
    peft_config=peft_config
)


#this code is just placeholder
if not (os.path.exists(CHECKPOINT_DIR) and os.listdir(CHECKPOINT_DIR)):
    # mark_only_lora_as_trainable(model)
    trainer.train()
    trainer.model.save_pretrained(MODEL_DIR)
    #.pt file


else:
    # Load the checkpoint with the correct config
    pmodel = PeftModel.from_pretrained(
        model,
        "checkpoints/checkpoint-10",
        is_trainable=True
    )

    # Now merge the weights properly with 16-bit precision to avoid rounding errors
    merged_model = pmodel.merge_and_unload(safe_merge=True)

    # Save the merged model
    save_path = "models/Text-Text-generation"
    merged_model.save_pretrained(save_path, safe_serialization=True, save_adapters=True, save_embedding_layers=True)
    tokenizer.save_pretrained(save_path)



# # Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
#     temperature=0.75,
#     max_tokens=2000,
#     top_p=1,
#     callback_manager=callback_manager,
#     verbose=True,  # Verbose is required to pass to the callback manager
# )


class Finetune():
    def __init__(self) -> None:
        pass
    
    def tune(self,model,dataset):
        MODEL_DIR = WORKSPACE_DIR / "models" / "Text-Text-generation"