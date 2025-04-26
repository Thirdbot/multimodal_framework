import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from login.huggingface_login import HuggingFaceLogin
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map, disk_offload
from accelerate.utils import set_seed

# Define paths
WORKSPACE_DIR = Path(__file__).parent.absolute()
MODEL_DIR = WORKSPACE_DIR / "models" / "Text-Text-generation"
OFFLOAD_DIR = WORKSPACE_DIR / "offload"

def load_merged_model():
    # Check if model directory exists
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found at {MODEL_DIR}")
        
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU setup.")
        
    # Create offload directory if it doesn't exist
    OFFLOAD_DIR.mkdir(exist_ok=True)
        
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True
    )
    
    # Initialize model with empty weights
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    
    # Calculate device map
    max_memory = {0: "6GB", "cpu": "30GB"}
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=model._no_split_modules
    )
    
    # Load and dispatch model with disk offloading
    model = load_checkpoint_and_dispatch(
        model,
        MODEL_DIR,
        device_map=device_map,
        offload_folder=str(OFFLOAD_DIR),
        offload_state_dict=True,
        no_split_module_classes=model._no_split_modules
    )
    
    # Enable disk offloading
    disk_offload(model, OFFLOAD_DIR, device_map=device_map)
    
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=100, temperature=0.7, top_p=0.9):
    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the appropriate device
    for k, v in inputs.items():
        inputs[k] = v.to(next(model.parameters()).device)
    
    # Generate text
    with torch.amp.autocast(device_type=next(model.parameters()).device.type):
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_merged_model()
    print("Model and tokenizer loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Example usage
    while True:
        prompt = input("\nEnter your prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
            
        print("\nGenerating response...")
        generated_text = generate_text(prompt, model, tokenizer)
        print("\nGenerated text:")
        print(generated_text)

if __name__ == "__main__":
    main()



#What To do

#this just addon.
# create fetch for dataset stoff
#create json file store datasets properties
#create class to finetune model with its compatible dataset Multiple time and can be chain function

#main stuff
#create event loop for use input and model input (should recieve multiple input type data as sametime)
#create attention between event loop for filter unwanting data so runtime not interfere
#create embbeding and en-router-attention with de-router-attention (shared attention or embed)
#create function feed input from router to encoder_model or decoder model
#create function to display output by model output from router_attention
#####note output from model should be stream into input of model_input instead of user_input or its model input for inteferencing
#####the data should be on eventloop instead of model loop so crack that 1 bit llms