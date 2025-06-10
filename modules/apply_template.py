#create template from model

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path

def run_model():
    # Get the workspace directory
    workspace_dir = Path(__file__).parent.parent.absolute()
    model_path = workspace_dir / "custom_models" / "conversation-model" / "kyutai-helium-1-2b"
    # model_path = "kyutai/helium-1-2b"
    
    # Load model and tokenizer from custom path without quantization
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        torch_dtype=torch.float16,  # Use float16 for memory efficiency
        low_cpu_mem_usage=True
    )
    
    # Create a conversation
    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! How are you today?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking! How can I help you today?"},
        {"role": "user", "content": "Can you tell me about yourself?"}
    ]
    
    # Process the conversation using tokenizer's chat template
    inputs = tokenizer(
        tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True),
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    print("\nGenerating response...")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and print response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nModel Response:")
    print(response)

if __name__ == "__main__":
    print("Starting conversation demo...")
    run_model()






