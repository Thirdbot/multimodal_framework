import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#for create base model to be use for finetuning and customize architecture
from transformers import LlamaModel, LlamaConfig, AutoTokenizer, AutoConfig,ProcessorMixin,PretrainedConfig,PreTrainedModel,CLIPVisionModel,CLIPProcessor,AutoModelForCausalLM,AutoModel
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch
from pathlib import Path
import json
from transformers.image_utils import load_image
from transformers import BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Config classes
class ConversationConfig(PretrainedConfig):
    model_type = "conversation-model"
    architectures = ["ConversationModel"]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VisionConfig(PretrainedConfig):
    model_type = "vision-model"
    architectures = ["VisionModel"]
    
    def __init__(self, lang_embed_dim=1024, clip_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.lang_embed_dim = lang_embed_dim
        self.clip_dim = clip_dim

# Model classes
class ConversationModel(PreTrainedModel):
    config_class = ConversationConfig
    
    def __init__(self, config, base_model):
        super().__init__(config)
        self.model = base_model
        self.config = config
        
        # Copy PEFT attributes from base model
        if hasattr(base_model, 'peft_config'):
            self.peft_config = base_model.peft_config
        if hasattr(base_model, 'active_adapter'):
            self.active_adapter = base_model.active_adapter
        if hasattr(base_model, 'base_model'):
            self.base_model = base_model.base_model
        
        # Enable gradient checkpointing for memory efficiency
        self.supports_gradient_checkpointing = True
        self._is_gradient_checkpointing = False
    
    def get_target_modules(self):
        """Detect the model architecture and return appropriate LoRA target modules."""
        if hasattr(self.model, 'get_target_modules'):
            return self.model.get_target_modules()
            
        model_type = self.config.model_type.lower() if hasattr(self.config, 'model_type') else ""
        
        target_modules_map = {
            "gpt2": ["c_attn", "c_proj"],
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "opt": ["q_proj", "k_proj", "v_proj", "out_proj"],
            "bloom": ["query_key_value", "dense"],
            "t5": ["q", "k", "v", "o"],
            "bert": ["query", "key", "value", "output.dense"],
            "roberta": ["query", "key", "value", "output.dense"],
            "gpt_neox": ["query_key_value", "dense"],
            "falcon": ["query_key_value", "dense"],
            "mpt": ["Wqkv", "out_proj"],
            "baichuan": ["W_pack", "o_proj"],
            "chatglm": ["query_key_value", "dense"],
            "qwen": ["c_attn", "c_proj"],
            "phi": ["Wqkv", "out_proj"],
            "gemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "stablelm": ["q_proj", "k_proj", "v_proj", "o_proj"]
        }
        
        return target_modules_map.get(model_type, ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Forward pass for conversation model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        return outputs
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    @property
    def is_gradient_checkpointing(self) -> bool:
        """Whether gradient checkpointing is enabled."""
        return self._is_gradient_checkpointing
    
    @is_gradient_checkpointing.setter
    def is_gradient_checkpointing(self, value: bool):
        """Set gradient checkpointing state."""
        self._is_gradient_checkpointing = value
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the model."""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        self.is_gradient_checkpointing = True
        
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the model."""
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        self.is_gradient_checkpointing = False
        
    def _set_gradient_checkpointing(self, module, value=False):
        """Set gradient checkpointing for a specific module."""
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
            
    def enable_input_require_grads(self):
        """Enable input gradients - required for gradient checkpointing."""
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
            
    def save_pretrained(self, save_directory, **kwargs):
        """Save the model."""
        # Save PEFT configuration if it exists
        if hasattr(self.model, 'peft_config'):
            self.model.save_pretrained(save_directory, **kwargs)
        else:
            super().save_pretrained(save_directory, **kwargs)

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

class VisionModel(PreTrainedModel):
    config_class = VisionConfig
    def __init__(self, config, vision_model, lang_model):
        super().__init__(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_model = vision_model
        self.vision_adapter = VisionAdapter(config.lang_embed_dim, config.clip_dim)
        self.lang_model = lang_model
        self.supports_gradient_checkpointing = True
        self._is_gradient_checkpointing = False
        
        embed_dim = self.lang_model.model.embed_tokens.weight.shape[1]

        self.text_adapter = torch.nn.Linear(embed_dim, config.lang_embed_dim).to(device)
        


    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                attend_to_img_tokens=True, labels=None, **kwargs):
        """Forward pass with proper loss calculation."""
        # Process inputs and get embeddings
        embeddings, attention_mask = self.process_inputs(input_ids, attention_mask, pixel_values, attend_to_img_tokens)

        # Pad labels to match the sequence length of embeddings
        if labels is not None:
            num_img_tokens = embeddings.shape[1] - input_ids.shape[1]  # Calculate image token count
            labels = torch.cat([
                torch.full((labels.shape[0], num_img_tokens), -100, dtype=labels.dtype, device=labels.device),
                labels
            ], dim=1)

        # Forward pass through language model
        outputs = self.lang_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        # If we have labels but no loss, calculate it
        if labels is not None and not hasattr(outputs, 'loss'):
            # Get logits from the output
            logits = outputs.last_hidden_state
            
            # Add gradient clipping to prevent explosion
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Warning: NaN or Inf detected in logits")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Ensure valid labels
            if torch.isnan(shift_labels).any() or torch.isinf(shift_labels).any():
                print(f"Warning: NaN or Inf detected in labels")
                shift_labels = torch.nan_to_num(shift_labels, nan=-100)
            
            # Calculate loss using CrossEntropyLoss with label smoothing
            loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
            
            # Reshape with safety checks
            try:
                vocab_size = shift_logits.size(-1)
                shift_logits_view = shift_logits.view(-1, vocab_size)
                shift_labels_view = shift_labels.view(-1)
                
                # Verify shapes before loss calculation
                if shift_logits_view.size(0) != shift_labels_view.size(0):
                    print(f"Shape mismatch: logits {shift_logits_view.shape}, labels {shift_labels_view.shape}")
                    raise ValueError("Logits and labels shape mismatch")
                
                loss = loss_fct(shift_logits_view, shift_labels_view)
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN or Inf loss detected, using mean reduction")
                    loss = loss_fct(shift_logits_view, shift_labels_view.clamp(min=0, max=vocab_size-1))
                
            except Exception as e:
                print(f"Error in loss calculation: {str(e)}")
                # Fallback to simple mean loss
                loss = torch.mean(shift_logits_view) * 0.0  # Zero loss to prevent NaN propagation
            
            # Create a new output object with loss
            outputs = CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
                hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
                cross_attentions=outputs.cross_attentions if hasattr(outputs, 'cross_attentions') else None,
            )
        
        return outputs
    
    
    @property
    def is_gradient_checkpointing(self) -> bool:
        """Whether gradient checkpointing is enabled."""
        return self._is_gradient_checkpointing
    
    @is_gradient_checkpointing.setter
    def is_gradient_checkpointing(self, value: bool):
        """Set gradient checkpointing state."""
        self._is_gradient_checkpointing = value
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the model."""
        if hasattr(self.vision_model, "gradient_checkpointing_enable"):
            self.vision_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        self.is_gradient_checkpointing = True
        
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the model."""
        if hasattr(self.vision_model, "gradient_checkpointing_disable"):
            self.vision_model.gradient_checkpointing_disable()
        self.is_gradient_checkpointing = False
        
    def _set_gradient_checkpointing(self, module, value=False):
        """Set gradient checkpointing for a specific module."""
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
            
    def enable_input_require_grads(self):
        """Enable input gradients - required for gradient checkpointing."""
        if hasattr(self.vision_model, "enable_input_require_grads"):
            self.vision_model.enable_input_require_grads()

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
        
    def process_inputs(self, input_ids, attention_mask, pixel_values, attend_to_img_tokens=True):
        # Processing inputs
        # device = torch.device("cuda")
                

        # In process_inputs:
        embeddings = self.lang_model.model.embed_tokens(input_ids)
        if self.text_adapter is not None:
            embeddings = self.text_adapter(embeddings)
        
        # #temporary cast dim
        # if embeddings.shape[-1] != 2048:
        #     embeddings = self.text_adapter(embeddings)

        device = next(self.lang_model.parameters()).device
        embeddings = embeddings.to(device)
        attention_mask = attention_mask.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

        if pixel_values is not None:
            image_embeddings = self.vision_model(pixel_values).last_hidden_state
            
            #clip dims to lang dims
            adapted_embeddings = self.vision_adapter(image_embeddings)
            embeddings = torch.cat((adapted_embeddings, embeddings), axis=1)
            attention_mask = self.__extend_attention_mask(attention_mask, attend_to_img_tokens)

        return embeddings, attention_mask
        
    def __extend_attention_mask(self, atten_mask, atten_to_img=True, num_added_tokens=257):
        # Extending the attention mask to image embeddings
        batch_size, seq_length = atten_mask.shape
        extended_mask = torch.ones if atten_to_img else torch.zeros
        mask = extended_mask((batch_size, seq_length + num_added_tokens),
                             dtype=atten_mask.dtype,
                             device=atten_mask.device)
        mask[:, -seq_length:] = atten_mask
        return mask

        
# Processor class
class VisionProcessor(ProcessorMixin):
    #added audio cause error needed audio
    attributes = ['image_processor','tokenizer','audio_tokenizer']

    def __init__(self,image_processor,tokenizer,audio_tokenizer=None):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.chat_template = self.tokenizer.chat_template
        self.audio_tokenizer = audio_tokenizer
        self.current_processor = self
    
    def add_label(self, inputs):
        num_images_tokens = 257  # Number of image tokens

        # Prepend -100 to labels for image tokens
        inputs['labels'] = torch.cat([
            torch.full((inputs['input_ids'].size(0), num_images_tokens), 
                       -100, dtype=inputs['input_ids'].dtype,
                       device=inputs['input_ids'].device),
            inputs['input_ids']
        ], dim=1)

        return inputs
    
    def __call__(self, text=None, images=None, create_labels=True):
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

# Register this config type when instantiated
AutoConfig.register("conversation-model", ConversationConfig)
# Register the config at class level
AutoConfig.register("vision-model", VisionConfig)
# Register this model type when instantiated
AutoModelForCausalLM.register(ConversationConfig, ConversationModel)
# Register the model at class level
AutoModelForCausalLM.register(VisionConfig, VisionModel)
    
class CreateModel:
    def __init__(self, model_name, model_category):
        self.model_name = model_name
        self.save_name = self.model_name.replace("/","_")
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
        
        # Load the original model and its config with more stable quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,  # Use float32 for compute
            bnb_4bit_quant_type="fp4",  # Use fp4 instead of nf4 for more stability
            bnb_4bit_use_double_quant=False,  # Disable double quantization
            llm_int8_threshold=0.0,  # Disable int8 threshold
            llm_int8_has_fp16_weight=False,
        )
        
        try:
            print(f"Loading model with stable quantization settings...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float32,  # Use float32 instead of bfloat16
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print(f"Model loaded successfully")
        except Exception as e:
            print(f"Error loading model with quantization: {str(e)}")
            print(f"Attempting to load without quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        self.original_config = AutoConfig.from_pretrained(self.model_name)
        
        # First load the tokenizer to get the correct vocab size
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            trust_remote_code=True
        )
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
        self.vision_processor = VisionProcessor(self.clip_processor, self.tokenizer)
        
        self.vision_config = VisionConfig(lang_embed_dim=1024,
                                          clip_dim=1024)
        
        self.tokenizer.chat_template = self.chat_template

    def add_conversation(self):
        """Add conversation capability to the model."""
        try:
            # Configure quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=False
            )
            
            # Load base model with quantization
            if not isinstance(self.model, AutoModelForCausalLM):
                print(f"Converting model to AutoModelForCausalLM")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float32
                )
            
            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Get target modules based on model architecture
            model_type = self.model.config.model_type.lower() if hasattr(self.model.config, 'model_type') else ""
            target_modules_map = {
                "gpt2": ["c_attn", "c_proj"],
                "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "opt": ["q_proj", "k_proj", "v_proj", "out_proj"],
                "bloom": ["query_key_value", "dense"],
                "t5": ["q", "k", "v", "o"],
                "bert": ["query", "key", "value", "output.dense"],
                "roberta": ["query", "key", "value", "output.dense"],
                "gpt_neox": ["query_key_value", "dense"],
                "falcon": ["query_key_value", "dense"],
                "mpt": ["Wqkv", "out_proj"],
                "baichuan": ["W_pack", "o_proj"],
                "chatglm": ["query_key_value", "dense"],
                "qwen": ["c_attn", "c_proj"],
                "phi": ["Wqkv", "out_proj"],
                "gemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "stablelm": ["q_proj", "k_proj", "v_proj", "o_proj"]
            }
            target_modules = target_modules_map.get(model_type, ["q_proj", "k_proj", "v_proj", "o_proj"])
            print(f"Using target modules for {model_type}: {target_modules}")
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=8,  # Rank
                lora_alpha=16,  # Alpha scaling
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Get PEFT model
            self.model = get_peft_model(self.model, lora_config)
            
            # Create conversation model
            self.convomodel = ConversationModel(self.original_config, self.model)
            
            # Disable caching for gradient checkpointing
            self.model.config.use_cache = False
            self.convomodel.config.use_cache = False
            
            # Enable training mode and gradient checkpointing
            self.convomodel.train()
            self.convomodel.gradient_checkpointing_enable()
            
            print(f"Successfully created conversation model with LoRA configuration")
            
            # Print trainable parameters
            trainable_params = 0
            all_param = 0
            for _, param in self.convomodel.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_param:.2f}%)")
            print(f"All params: {all_param:,}")
            
        except Exception as e:
            print(f"Error creating conversation model: {str(e)}")
            raise
    
    def add_vision(self):
        # Initialize models and processor with quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        self.vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        self.vismodel = VisionModel(self.vision_config, self.vision_model, self.model)
        
    def save_regular_model(self):
        """Save the model and all its components with optimizations."""
        try:
            # Create necessary directories
            os.makedirs(self.model_path, exist_ok=True)
            
            # Save the model with safe serialization
            self.convomodel.save_pretrained(
                self.model_path,
                safe_serialization=True
            )
            
            # Save tokenizer
            self.tokenizer.save_pretrained(self.model_path)
            
            # Save config
            self.original_config.save_pretrained(self.model_path)
            
            print(f"Successfully saved model to {self.model_path}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise
        
    def save_vision_model(self):
        """Save the model and all its components with optimizations."""
        # Create necessary directories
        vision_model_path = os.path.join(self.model_path, "vision_model")
        lang_model_path = os.path.join(self.model_path, "lang_model")
        os.makedirs(vision_model_path, exist_ok=True)
        os.makedirs(lang_model_path, exist_ok=True)
        
        # Save base model first with quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        # Save tokenizer
        self.tokenizer.save_pretrained(
            self.model_path,
            legacy_format=False
        )
        
       
        
        # Save vision model with quantization
        self.vision_model.save_pretrained(
            vision_model_path,
            # self.model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            safe_serialization=True
        )
        
        # Save language model with quantization
        self.model.save_pretrained(
            lang_model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            safe_serialization=True
        )
        
        # Save vision processor
        self.vision_processor.save_pretrained(
            os.path.join(self.model_path, "vision_processor"),
            safe_serialization=True
        )
        
        # Save main model configuration
        self.vision_config.save_pretrained(self.model_path)
        
        
        
        # # Create Modelfile for Ollama
        # modelfile_content = f"""FROM python:3.9

        # # Set up the model directory
        # WORKDIR /model
        # COPY . /model/

        # # Install required packages
        # RUN pip install transformers torch torchvision pillow sentencepiece protobuf

        # # Set up the model configuration
        # PARAMETER temperature 0.7
        # PARAMETER top_p 0.9
        # PARAMETER top_k 40
        # PARAMETER num_ctx 2048

        # # Set up the model template
        # TEMPLATE \"\"\"{self.chat_template}\"\"\"

        # # Set up the model system prompt
        # SYSTEM \"\"\"You are a helpful AI assistant that can understand and describe images.\"\"\"

        # # Set up the model parameters
        # PARAMETER stop "Human:"
        # PARAMETER stop "Assistant:"

        # # Set up model architecture
        # PARAMETER model_type "vision-model"
        # PARAMETER clip_dim {self.vision_config.clip_dim}
        # PARAMETER lang_embed_dim {self.vision_config.lang_embed_dim}
        # """
        
        # # Save Modelfile
        # with open(os.path.join(self.model_path, "Modelfile"), "w") as f:
        #     f.write(modelfile_content)
            
 


# Load model and processor from demo_path
def load_saved_model(model_path,checkpoint=False):
    """Load a saved model and its processor."""    
    try:
        # Load the config
        config = AutoConfig.from_pretrained(model_path)
        print(f"Loaded config with model type: {config.model_type}")
        print(f"Model architecture: {config.architectures}")
        
        # Check if this is a vision model
        is_vision_model = (
            hasattr(config, 'model_type') and config.model_type == "vision-model" or
            hasattr(config, 'architectures') and config.architectures and "VisionModel" in config.architectures
        )
        
        if is_vision_model and checkpoint:
            vision_model_path = os.path.join(model_path, "vision_model")
            lang_model_path = os.path.join(model_path, "lang_model")
            
        if is_vision_model and not checkpoint:
            vision_model_path = os.path.join(model_path, "vision_model")
            lang_model_path = os.path.join(model_path, "lang_model")

        if is_vision_model:
            print("Loading vision model...")
            
            # Load components
            vision_model = CLIPVisionModel.from_pretrained(vision_model_path)
            lang_model = AutoModelForCausalLM.from_pretrained(lang_model_path,torch_dtype=torch.float32)
            
            # Create model
            model = VisionModel(config, vision_model, lang_model)
            
            # Load processor
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            model.config.use_cache = False
            model.train()  # Ensure model is in training mode
            return model, tokenizer
        else:
            # For conversation models, use AutoModelForCausalLM
            print("Loading conversation model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            # Create conversation model
            model = ConversationModel(config, base_model)
            model.config.use_cache = False
            model.train()  # Ensure model is in training mode
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

     
# #create template from model

# Example usage:

# Load the model and tokenizer

# Load the model and tokenizer
# path = Path(__file__).parent.parent.absolute() / "custom_models" / "vision-model" / "Qwen_Qwen1.5-0.5B-Chat"
# model, tokenizer = load_saved_model(path.as_posix())

# # Get the device and dtype of the model
# device = next(model.parameters()).device
# dtype = next(model.parameters()).dtype

# # Load the image
# image_url = "https://media.istockphoto.com/id/155439315/photo/passenger-airplane-flying-above-clouds-during-sunset.jpg?s=612x612&w=0&k=20&c=LJWadbs3B-jSGJBVy9s0f8gZMHi2NvWFXa3VJ2lFcL0="
# image = load_image(image_url)

# # Initialize the processor
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
# processor = VisionProcessor(image_processor=clip_processor, tokenizer=tokenizer)

# # Tokenize the text
# text_input = processor.tokenizer("What do you see in this image?", return_tensors="pt")

# # Preprocess the image
# pixel_values = processor.image_processor(images=image, return_tensors="pt")["pixel_values"]

# # Move inputs to the model's device and dtype
# inputs = {
#     "input_ids": text_input["input_ids"].to(device),  # Keep input_ids as integers
#     "attention_mask": text_input["attention_mask"].to(device).to(dtype),
#     "pixel_values": pixel_values.to(device).to(dtype)
# }

# # Generate outputs
# outputs = model.generate(
#     input_ids=inputs["input_ids"],
#     attention_mask=inputs["attention_mask"],
#     pixel_values=inputs["pixel_values"],
#     max_new_tokens=50,  # Limit the number of tokens to generate
#     num_beams=4,        # Use beam search for better results
#     temperature=0.7,    # Sampling temperature
#     do_sample=True      # Enable sampling
# )

# # Decode and print the outputs
# decoded_outputs = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
# print(decoded_outputs)




# Working Good
# Load the model and tokenizer
# path = Path(__file__).parent.parent.absolute() / "custom_models" / "conversation-model" / "Qwen_Qwen1.5-0.5B-Chat"
# model, tokenizer = load_saved_model(path.as_posix())

# # Move the model to the appropriate device
# device = next(model.parameters()).device

# # Define the text input
# text_prompt = "Nigga"

# # Tokenize the text input
# text_input = tokenizer(text_prompt, return_tensors="pt").to(device)

# # Generate outputs
# outputs = model.generate(
#     input_ids=text_input["input_ids"],
#     attention_mask=text_input["attention_mask"],
#     max_new_tokens=50,  # Limit the number of tokens to generate
#     num_beams=4,        # Use beam search for better results
#     temperature=0.7,    # Sampling temperature
#     do_sample=True      # Enable sampling
# )

# # Decode and print the outputs
# decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# print(decoded_outputs)
