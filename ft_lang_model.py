
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,LoraConfig
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#model
tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/WizardLM-13B-Uncensored")
model = AutoModelForCausalLM.from_pretrained("cognitivecomputations/WizardLM-13B-Uncensored")


#fine-tune
peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
peft_model = PeftModel.from_pretrained(model,peft_config=peft_config)


merged_model = peft_model.merge_and_unload()

merged_model.save_pretrained("merged_model", safe_serialization=True)


