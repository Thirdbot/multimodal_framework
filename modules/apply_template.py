#create template from model

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import pipeline

# Define a multimodal chat template
chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}
{{ message['content'] }}
{% elif message['role'] == 'user' %}
Human: {% if message.get('images') %}
[Images: {{ message['images']|length }}]
{% endif %}
{{ message['content'] }}
{% elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}
{% endif %}
{% endfor %}
Assistant:"""

def load_image(image_path_or_url):
    """Load image from path or URL."""
    if image_path_or_url.startswith(('http://', 'https://')):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_url)
    return image

def get_image_description(image):
    """Get a text description of the image using a vision-language model."""
    # Load a vision-language model for image captioning
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    # Generate description
    description = image_to_text(image)[0]['generated_text']
    return description

def run_multimodal_demo():
    # Load the text model
    model_name = "kyutai/helium-1-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Example images
    image_urls = [
        "https://cdn.britannica.com/34/180334-138-4235A017/subordinate-meerkat-pack.jpg?w=800&h=450&c=crop",
        "https://static.vecteezy.com/system/resources/thumbnails/002/098/203/small_2x/silver-tabby-cat-sitting-on-green-background-free-photo.jpg"
    ]
    
    # Load and describe images
    print("Loading and analyzing images...")
    images = [load_image(url) for url in image_urls]
    image_descriptions = [get_image_description(img) for img in images]
    
    # Create a conversation that includes image descriptions
    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant that can understand images through their descriptions."},
        {
            "role": "user", 
            "content": f"I have some images to show you. Here are their descriptions:\n" + 
                      "\n".join([f"Image {i+1}: {desc}" for i, desc in enumerate(image_descriptions)]) +
                      "\nWhat can you tell me about these images?"
        },
        {
            "role": "assistant", 
            "content": "I can see these are images of meerkats in their natural habitat."
        },
        {
            "role": "user", 
            "content": f"Can you describe the first image in more detail?\nImage 1: {image_descriptions[0]}"
        }
    ]
    
    # Process the conversation
    tokenizer.chat_template = chat_template
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
            top_p=0.9
        )
    
    # Decode and print response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nModel Response:")
    print(response)

if __name__ == "__main__":
    print("Starting multimodal demo...")
    run_multimodal_demo()






