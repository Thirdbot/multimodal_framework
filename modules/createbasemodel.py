import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#for create base model to be use for finetuning and customize architecture
from transformers import LlamaModel, LlamaConfig, AutoTokenizer, AutoConfig,ProcessorMixin,PretrainedConfig,PreTrainedModel,CLIPVisionModel,CLIPProcessor,AutoModelForCausalLM,AutoModel
from transformers import AutoProcessor
import torch
from pathlib import Path
import json
from transformers.image_utils import load_image



       
# #create template from model

class VisionProcessor(ProcessorMixin):
    attributes = ['image_processor','tokenizer']
    
    def __init__(self,image_processor,tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.chat_template = self.tokenizer.chat_template
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
    model_type = "multimodal"  # Changed from "vision_language_model" to "multimodal"
    def __init__(self, lang_embed_dim=2048, clip_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.lang_embed_dim = lang_embed_dim
        self.clip_dim = clip_dim

class VisionModel(PreTrainedModel):
    config_class = VisionConfig
    
    def __init__(self,config,vision_model,lang_model):
        super().__init__(config,vision_model,lang_model)
        self.vision_model = vision_model
        self.vision_adapter = VisionAdapter(config.lang_embed_dim,config.clip_dim)
        self.lang_model = lang_model
        
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



#goal is to make a tokenizer and a model and save it to the models directory

#simple model backbone using llama (customize later)
class CreateModel:
    def __init__(self, model_name, model_category):
        self.model_name = model_name
        self.save_name = self.model_name.replace("/","-")
        self.model_category = model_category
        
        self.chat_template = """{% for message in messages %}
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
        
        self.model_path = Path(__file__).parent.parent.absolute() / "custom_models" / self.model_category / self.save_name
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self.lang_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        # First load the tokenizer to get the correct vocab size
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )
        
        self.tokenizer.chat_template = self.chat_template
        
        # Get the original config to maintain compatibility
        self.original_config = AutoConfig.from_pretrained(self.model_name)
        
        # Create modified config based on original
        self.BaseLlamaConfig = LlamaConfig(
            vocab_size=len(self.tokenizer),  # Use actual vocab size
            hidden_size=2048,
            num_hidden_layers=16,
            num_attention_heads=16,
            intermediate_size=5504,
            max_position_embeddings=self.original_config.max_position_embeddings,  # Keep original context length
            torch_dtype=torch.float16,
            use_cache=False,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Initialize model with memory optimizations
        self.model = LlamaModel(
            self.BaseLlamaConfig
            )
        

    def add_vision(self):
        # Initialize models and processor
        self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_processor = VisionProcessor(self.clip_processor, self.tokenizer)
        
        self.config = VisionConfig()
        self.model = VisionModel(self.config, self.vision_model, self.model)
        
    def save_regular_model(self):
        """Save the model and all its components with optimizations."""
        # Create necessary directories
        lang_model_path = os.path.join(self.model_path, "lang_model")
        os.makedirs(lang_model_path, exist_ok=True)
        
        # Save language model
        self.lang_model.save_pretrained(lang_model_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.model_path)
        
        
        
    def save_vision_model(self):
        """Save the model and all its components with optimizations."""
        # Create necessary directories
        vision_model_path = os.path.join(self.model_path, "vision_model")
        lang_model_path = os.path.join(self.model_path, "lang_model")
        os.makedirs(vision_model_path, exist_ok=True)
        os.makedirs(lang_model_path, exist_ok=True)
        
        # Save vision model and processor
        self.vision_model.save_pretrained(vision_model_path)
        self.clip_processor.save_pretrained(vision_model_path)
        
        # Save language model
        self.lang_model.save_pretrained(lang_model_path)
        
        # Save vision processor
        self.vision_processor.save_pretrained(os.path.join(self.model_path, "vision_processor"))
        
        # Save main model configuration
        self.config.save_pretrained(self.model_path)
        
        # Save main model
        self.model.save_pretrained(
            self.model_path,
            max_shard_size="500MB",
            safe_serialization=True
        )
        
        # Save tokenizer
        self.tokenizer.save_pretrained(
            self.model_path,
            legacy_format=False
        )
        
        # Create Modelfile for Ollama
        modelfile_content = f"""FROM python:3.9

        # Set up the model directory
        WORKDIR /model
        COPY . /model/

        # Install required packages
        RUN pip install transformers torch torchvision pillow sentencepiece protobuf

        # Set up the model configuration
        PARAMETER temperature 0.7
        PARAMETER top_p 0.9
        PARAMETER top_k 40
        PARAMETER num_ctx 2048

        # Set up the model template
        TEMPLATE \"\"\"{self.chat_template}\"\"\"

        # Set up the model system prompt
        SYSTEM \"\"\"You are a helpful AI assistant that can understand and describe images.\"\"\"

        # Set up the model parameters
        PARAMETER stop "Human:"
        PARAMETER stop "Assistant:"

        # Set up model architecture
        PARAMETER model_type "multimodal"
        PARAMETER clip_dim {self.config.clip_dim}
        PARAMETER lang_embed_dim {self.config.lang_embed_dim}
        """
        
        # Save Modelfile
        with open(os.path.join(self.model_path, "Modelfile"), "w") as f:
            f.write(modelfile_content)
 


created_model = CreateModel("kyutai/helium-1-2b","text-generation")
# created_model.add_vision()
# created_model.save_vision_model()

demo_path = Path(__file__).parent.parent.absolute() / "custom_models" / "text-generation" / "kyutai-helium-1-2b"


# Register at module level before any class definitions
AutoConfig.register("multimodal", VisionConfig)
AutoModel.register(VisionModel, VisionConfig)

# Load model and processor from demo_path
def load_saved_model(model_path):
    """Load a saved model and its processor."""
    demo_path = model_path
    
    # Load the config
    config = AutoConfig.from_pretrained(demo_path)
    print(f"Loaded config with model type: {config.model_type}")
    print(f"Model architecture: {config.architectures}")
    
    # Load the model using AutoModel with the config
    if config.model_type == "multimodal" and config.architectures[0] == "VisionModel":
        vision_model_path = os.path.join(demo_path, "vision_model")
        lang_model_path = os.path.join(demo_path, "lang_model")
        
        vision_model = CLIPVisionModel.from_pretrained(vision_model_path)
        lang_model = AutoModelForCausalLM.from_pretrained(lang_model_path)
        
        model = VisionModel(config, vision_model, lang_model)
        
        # Load the vision processor components
        image_processor = CLIPProcessor.from_pretrained(vision_model_path)
        tokenizer = AutoTokenizer.from_pretrained(demo_path)
        processor = VisionProcessor(image_processor=image_processor, tokenizer=tokenizer)
    else:
        model = AutoModel.from_pretrained(
            demo_path,
            config=config,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(demo_path)
    
    return model, processor

# Example usage:
# model, processor = load_saved_model()
# image = load_image("https://media.istockphoto.com/id/155439315/photo/passenger-airplane-flying-above-clouds-during-sunset.jpg?s=612x612&w=0&k=20&c=LJWadbs3B-jSGJBVy9s0f8gZMHi2NvWFXa3VJ2lFcL0=")
# inputs = processor(text="What do you see in this image?", images=image)
# outputs = model.generate(**inputs)
# print(processor.tokenizer.batch_decode(outputs, skip_special_tokens=False))
        