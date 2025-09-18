# #create template from model

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads

from transformers import AutoModelForCausalLM,ProcessorMixin,PretrainedConfig,CLIPVisionModel,CLIPProcessor,AutoTokenizer, AutoConfig,AutoModel
from transformers.image_utils import load_image
from transformers.modeling_utils import PreTrainedModel
import torch
import torch.nn as nn
# from PIL import Image
from pathlib import Path
import requests
from io import BytesIO
import base64


class VisionProcessor(ProcessorMixin):
    attributes = ['image_processor','tokenizer']
    
    def __init__(self,image_processor,tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.current_processor = self
    
    #create place for images in the input_ids
    def add_label(self,inputs):
        special_images_tokens = self.tokenizer.convert_tokens_to_ids("<image>")
        num_images_tokens = 257
        
        input_ids = inputs['input_ids'].clone()
        special_images_column = torch.full(
            (input_ids.size(0),num_images_tokens),
            float(special_images_tokens),
            device=input_ids.device,
            dtype=input_ids.dtype
        )
        
        inputs['labels'] = torch.cat([special_images_column,input_ids],dim=1)
        
        return inputs
    
    def __call__(self, text=None, images=None, create_labels=False):
        """
        Process images and/or text inputs.
        
        Args:
            text: Text input to process
            images: Image input to process
            create_labels: If True, creates labels for training
        """
        result = {}

        # Process images
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            # Use CLIP processor's image processing
            result["pixel_values"] = self.image_processor(images=images, return_tensors="pt")["pixel_values"]

        # Process text
        if text is not None:
            text_result = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=250,
                return_tensors="pt"
            )
            result["input_ids"] = text_result["input_ids"]
            result["attention_mask"] = text_result["attention_mask"]

        # Optionally create labels
        if create_labels and "input_ids" in result:
            result = self.add_label(result)

        return result

class VisionAdapter(torch.nn.Module):
    def __init__(self, lang_embed_dim, clip_dim):
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.layer1 = torch.nn.Linear(clip_dim, 500)
        self.layer2 = torch.nn.Linear(500, 500)
        self.layer3 = torch.nn.Linear(500, lang_embed_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        output = self.activation(x)
        return output


class VisionConfig(PretrainedConfig):
    model_type = "vision-model"
    architectures = ["VisionModel"]
    def __init__(self, lang_embed_dim=2048, clip_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.lang_embed_dim = lang_embed_dim
        self.clip_dim = clip_dim
        # Register this config type when instantiated
        AutoConfig.register("vision-model", VisionConfig)

class VisionModel(PreTrainedModel):
    config_class = VisionConfig
    
    def __init__(self, config, vision_model, lang_model):
        super().__init__(config)  # Only pass config to parent
        self.vision_model = vision_model
        self.vision_adapter = VisionAdapter(config.lang_embed_dim, config.clip_dim)
        self.lang_model = lang_model
        # Register this model type when instantiated
        AutoModel.register(VisionConfig, VisionModel)
        
    def __extend_attention_mask(self, atten_mask, atten_to_img=True, num_added_tokens=257):
        # Extending the attention mask to image embeddings
        batch_size, seq_length = atten_mask.shape
        extended_mask = torch.ones if atten_to_img else torch.zeros
        mask = extended_mask((batch_size, seq_length + num_added_tokens),
                             dtype=atten_mask.dtype,
                             device=atten_mask.device)
        mask[:, -seq_length:] = atten_mask
        return mask
    def process_inputs(self, input_ids, attention_mask, pixel_values, attend_to_img_tokens=True):
        # Processing inputs
        embeddings = self.lang_model.model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_embeddings = self.vision_model(pixel_values).last_hidden_state
            adapted_embeddings = self.vision_adapter(image_embeddings)
            embeddings = torch.cat((adapted_embeddings, embeddings), axis=1)
            attention_mask = self.__extend_attention_mask(attention_mask, attend_to_img_tokens)

        return embeddings, attention_mask

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                attend_to_img_tokens=True, labels=None, **kwargs):
        input_ids = kwargs.get("input_ids", input_ids)
        attention_mask = kwargs.get("attention_mask", attention_mask)
        pixel_values = kwargs.get("pixel_values", pixel_values)
        labels = kwargs.get("labels", labels)

        embeddings, attention_mask = self.process_inputs(input_ids, attention_mask, pixel_values, attend_to_img_tokens)

        outputs = self.lang_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs

    def generate(self, input_ids=None, attention_mask=None, pixel_values=None,
                 attend_to_img_tokens=True, **kwargs):
        input_ids = kwargs.pop("input_ids", input_ids)
        attention_mask = kwargs.pop("attention_mask", attention_mask)
        pixel_values = kwargs.pop("pixel_values", pixel_values)

        embeddings, attention_mask = self.process_inputs(input_ids, attention_mask, pixel_values, attend_to_img_tokens)

        if "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = 100
        if "min_length" not in kwargs:
            kwargs["min_length"] = 1
        if "num_beams" not in kwargs:
            kwargs["num_beams"] = 4
        if "temperature" not in kwargs:
            kwargs["temperature"] = 0.7
        if "do_sample" not in kwargs:
            kwargs["do_sample"] = True

        return self.lang_model.generate(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            **kwargs
        )

# Initialize models and processor
vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
lang_model = AutoModelForCausalLM.from_pretrained("kyutai/helium-1-2b")
tokenizer = AutoTokenizer.from_pretrained("kyutai/helium-1-2b")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

vision_processor = VisionProcessor(clip_processor, tokenizer)
config = VisionConfig()
model = VisionModel(config, vision_model, lang_model)

print("\n=== Model Architecture Details ===")
print(f"Model :{model}")
print("\n1. Vision Model (CLIP):")
print(f"Model Type: {type(vision_model).__name__}")
print(f"Vision Config: {vision_model.config}")  
print(f"Number of Parameters: {sum(p.numel() for p in vision_model.parameters())}")

print("\n2. Language Model (Helium):")
print(f"Model Type: {type(lang_model).__name__}")
print(f"Model Config: {lang_model.config}")
print(f"Number of Parameters: {sum(p.numel() for p in lang_model.parameters())}")

print("\n3. Vision Adapter:")
print(f"Input Dimension (CLIP): {config.clip_dim}")
print(f"Output Dimension (Language): {config.lang_embed_dim}")
print(f"Adapter Architecture: {model.vision_adapter}")

print("\n4. Tokenizer Information:")
print(f"Tokenizer Type: {type(tokenizer).__name__}")
print(f"Vocabulary Size: {tokenizer.vocab_size}")
print(f"Model Max Length: {tokenizer.model_max_length}")
print(f"Special Tokens: {tokenizer.special_tokens_map}")

print("\n5. CLIP Processor Information:")
print(f"Processor Type: {type(clip_processor).__name__}")
print(f"Image Size: {clip_processor.image_processor.size}")
print(f"Image Mean: {clip_processor.image_processor.image_mean}")
print(f"Image Std: {clip_processor.image_processor.image_std}")


# Test the model
image = load_image("https://media.istockphoto.com/id/155439315/photo/passenger-airplane-flying-above-clouds-during-sunset.jpg?s=612x612&w=0&k=20&c=LJWadbs3B-jSGJBVy9s0f8gZMHi2NvWFXa3VJ2lFcL0=")
text = "What do you see in this image?"

inputs = vision_processor(text=text, images=image)
outputs = model.generate(**inputs, max_new_tokens=100)
print("\n=== Generation Output ===")
print(vision_processor.tokenizer.batch_decode(outputs, skip_special_tokens=True))

        