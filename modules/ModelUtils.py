import os
import torch
import json
from pathlib import Path
from dataclasses import dataclass

from transformers import (
    AutoTokenizer, AutoConfig, ProcessorMixin, PreTrainedModel,
    CLIPProcessor, AutoModelForCausalLM, BitsAndBytesConfig
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training,PeftModel
from transformers import CLIPVisionModel

from modules.variable import Variable
from modules.ModelCreationtemplate import ModelTemplate


from modules.models.ConversationModel import ConversationConfig,ConversationModel
from modules.models.VisionModel import VisionConfig

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

### add target modules base on model architecture like custom model has custom architecture
TARGET_MODULES_MAP = {
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
    "stablelm": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "ConversationModelWrapper": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "VisionModelWrapper": ["q_proj", "k_proj", "v_proj", "o_proj"],

}


def get_target_modules(model_type: str):
    """Get target modules for LoRA based on model type."""
    return TARGET_MODULES_MAP.get(model_type, None)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    load_in_4bit: bool = True
    compute_dtype: torch.dtype = torch.float32
    quant_type: str = "fp4"
    use_double_quant: bool = False
    llm_int8_threshold: float = 0.0
    llm_int8_has_fp16_weight: bool = False


@dataclass
class ModelConfig:
    """General model configuration."""
    clip_processor_name: str = "openai/clip-vit-large-patch14"
    use_fast_tokenizer: bool = True
    use_cache: bool = False
    gradient_checkpointing: bool = True
    quantization: QuantizationConfig | None = None


# class ConversationModelWrapper(PreTrainedModel):
#     config_class = ConversationConfig

#     def __init__(self, config, **kwargs):
#         # Extract base_model from kwargs for backward compatibility
#         base_model = kwargs.pop('base_model', None)
#         super().__init__(config, inner_model=base_model, **kwargs)

#     def get_target_modules(self):
#         if hasattr(self.model, 'get_target_modules'):
#             return self.model.get_target_modules()
#         model_type = self.config.model_type.lower() if hasattr(self.config, 'model_type') else ""
#         return get_target_modules(model_type)
    
#     def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
#         outputs = self.model(
#             input_ids=input_ids.to(self.device),
#             attention_mask=attention_mask.to(self.device),
#             labels=labels.to(self.device) if labels is not None else None,
#             **kwargs
#         )
#         return outputs
    
    # def save_pretrained(self, save_directory, **kwargs):
    #     """Save the inner model directly instead of wrapper."""
    #     if self.model is not None:
    #         # Save the wrapped model (which has the actual weights)
    #         self.model.save_pretrained(save_directory, **kwargs)
    #     else:
    #         # Fallback to parent save if no inner model
    #         super().save_pretrained(save_directory, **kwargs)

class ConversationModelWrapper(PreTrainedModel):
    config_class = ConversationConfig
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, config, base_model):
        super().__init__(config)
        self.bmodel = base_model.to(self.device)
        self.config = config
        
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bmodel(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=labels.to(self.device) if labels is not None else None,
            **kwargs
        )
        return outputs

    def generate(self, *args, **kwargs):
        return self.bmodel.generate(*args, **kwargs)

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self._is_gradient_checkpointing

    @is_gradient_checkpointing.setter
    def is_gradient_checkpointing(self, value: bool):
        self._is_gradient_checkpointing = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.bmodel, "gradient_checkpointing_enable"):
            self.bmodel.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        self.is_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.bmodel, "gradient_checkpointing_disable"):
            self.bmodel.gradient_checkpointing_disable()
        self.is_gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_input_require_grads(self):
        if hasattr(self.bmodel, "enable_input_require_grads"):
            self.bmodel.enable_input_require_grads()
    
    def named_parameters(self, *args, **kwargs):
        """Expose inner model's parameters so trainer sees them."""
        return self.bmodel.named_parameters(*args, **kwargs)
    
    def parameters(self, *args, **kwargs):
        """Expose inner model's parameters so trainer sees them."""
        return self.bmodel.parameters(*args, **kwargs)
    
    def train(self, mode=True):
        """Ensure inner model is in training mode."""
        super().train(mode)
        if self.bmodel is not None:
            self.bmodel.train(mode)
        return self
            
    def save_pretrained(self, save_directory, **kwargs):
        """Save the model."""
        # Save PEFT configuration if it exists
        if hasattr(self.bmodel, 'peft_config'):
            self.bmodel.save_pretrained(save_directory, **kwargs)
        else:
            super().save_pretrained(save_directory, **kwargs)


    

class VisionAdapter(torch.nn.Module):
    def __init__(self, lang_embed_dim, clip_dim):
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.layer1 = torch.nn.Linear(clip_dim, 500)
        self.layer2 = torch.nn.Linear(500, 1024)
        self.layer3 = torch.nn.Linear(1024, lang_embed_dim)

    def forward(self, x):
        # Ensure the input tensor matches the model's dtype
        x = x.to(self.layer1.weight.dtype)  # Match the dtype of the layer weights
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        output = self.activation(x)
        return output

class VisionModelWrapper(PreTrainedModel):
    config_class = VisionConfig

    def __init__(self, config, lang_model=None, model_config: ModelConfig | None = None):
        super().__init__(config)
        self.model_config = model_config or ModelConfig()
        self.vision_model = CLIPVisionModel.from_pretrained(self.model_config.clip_processor_name)
        self.vision_adapter = VisionAdapter(1024, 1024)
        
        # Freeze vision model but keep gradients through adapter
        for param in self.vision_model.parameters():
            param.requires_grad = False

        self.lang_model = lang_model
        self.supports_gradient_checkpointing = True
        self._is_gradient_checkpointing = False
        self.config = config
        
        # Ensure all components are on the same device as language model
        if self.lang_model is not None:
            device = next(self.lang_model.parameters()).device
            self.vision_model = self.vision_model.to(device)
            self.vision_adapter = self.vision_adapter.to(device)

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
        return self._is_gradient_checkpointing

    @is_gradient_checkpointing.setter
    def is_gradient_checkpointing(self, value: bool):
        self._is_gradient_checkpointing = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.vision_model, "gradient_checkpointing_enable"):
            self.vision_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        self.is_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.vision_model, "gradient_checkpointing_disable"):
            self.vision_model.gradient_checkpointing_disable()
        self.is_gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value


    def enable_input_require_grads(self):
        if hasattr(self.vision_model, "enable_input_require_grads"):
            self.vision_model.enable_input_require_grads()

    # def named_parameters(self, *args, **kwargs):
    #     """Expose inner model's parameters so trainer sees them."""
    #     return self.lang_model.named_parameters(*args, **kwargs)
    
    # def parameters(self, *args, **kwargs):
    #     """Expose inner model's parameters so trainer sees them."""
    #     return self.lang_model.parameters(*args, **kwargs)
    
    # def train(self, mode=True):
    #     """Ensure inner model is in training mode."""
    #     super().train(mode)
    #     if self.lang_model is not None:
    #         self.lang_model.train(mode)
    #     return self

    def generate(self, input_ids=None, attention_mask=None, pixel_values=None,
                 attend_to_img_tokens=True, **kwargs):
        input_ids = kwargs.pop("input_ids", input_ids)
        attention_mask = kwargs.pop("attention_mask", attention_mask)
        pixel_values = kwargs.pop("pixel_values", pixel_values)

        embeddings, attention_mask = self.process_inputs(input_ids, attention_mask, pixel_values, attend_to_img_tokens)

        kwargs.setdefault("max_new_tokens", 100)
        kwargs.setdefault("min_length", 1)
        kwargs.setdefault("num_beams", 4)
        kwargs.setdefault("temperature", 0.7)
        kwargs.setdefault("do_sample", True)

        return self.lang_model.generate(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            **kwargs
        )
        
    def process_inputs(self, input_ids, attention_mask, pixel_values, attend_to_img_tokens=True):
        # Get the device and dtype of the model
        device = next(self.lang_model.parameters()).device
        dtype = next(self.lang_model.parameters()).dtype

        # Move all inputs to the correct device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Move embeddings and attention mask to the correct device and dtype
        embeddings = self.lang_model.get_input_embeddings()(input_ids)
        embeddings = embeddings.to(device).to(dtype)
        attention_mask = attention_mask.to(device).to(dtype)

        if pixel_values is not None:
            pixel_values = pixel_values.to(device).to(dtype)
            # Ensure vision model and adapter are on the correct device
            self.vision_model = self.vision_model.to(device)
            self.vision_adapter = self.vision_adapter.to(device)
            
            # Process through vision model (frozen) and detach to ensure clean gradient flow through adapter
            with torch.no_grad():
                image_embeddings = self.vision_model(pixel_values).last_hidden_state.to(dtype).to(device)
            
            # Ensure adapter output has gradient tracking
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

        
class VisionProcessor(ProcessorMixin):
    attributes = ['image_processor', 'tokenizer']

    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.chat_template = self.tokenizer.chat_template
    
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
    
class CreateModel:
    def __init__(self, model_repo_path, model_category, model_config: ModelConfig | None = None):
        self.model_config = model_config or ModelConfig()
        self.model_repo_path = model_repo_path
        self.save_name = self.model_repo_path.name.replace("/", "_")
        self.model_category = model_category
        self.variable = Variable()
        self.dtype = self.variable.DTYPE
        self.model_path = Path(__file__).parent.parent.absolute() / "custom_models" / self.model_category / self.save_name
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Use configurable quantization
        quant_cfg = self.model_config.quantization or QuantizationConfig()
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg.load_in_4bit,
            bnb_4bit_compute_dtype=quant_cfg.compute_dtype,
            bnb_4bit_quant_type=quant_cfg.quant_type,
            bnb_4bit_use_double_quant=quant_cfg.use_double_quant,
            llm_int8_threshold=quant_cfg.llm_int8_threshold,
            llm_int8_has_fp16_weight=quant_cfg.llm_int8_has_fp16_weight,
        )

        try:
            print("Loading model with quantization settings...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_repo_path,
                quantization_config=self.quantization_config,
                device_map="auto",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print("Model loaded successfully")
        except Exception as e:
            print("Attempting to load without quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_repo_path,
                device_map="auto",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

        self.original_config = ConversationConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_repo_path,
            use_fast=self.model_config.use_fast_tokenizer,
            trust_remote_code=True
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            self.model_config.clip_processor_name,
            use_fast=self.model_config.use_fast_tokenizer
        )
        self.vision_processor = VisionProcessor(self.clip_processor, self.tokenizer)
        self.vision_config = VisionConfig()
    
    def add_conversation(self):
        """Add conversation capability to the model."""
        try:
            if not isinstance(self.model, AutoModelForCausalLM):
                print("Converting model to AutoModelForCausalLM")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_repo_path,
                    device_map="auto",
                    trust_remote_code=True,
                    quantization_config=self.quantization_config,
                    torch_dtype=self.dtype
                )
            
            self.model = prepare_model_for_kbit_training(self.model)
            
            
            # model_type = self.model.config.model_type.lower() if hasattr(self.model.config, 'model_type') else ""
            model_arc = self.model.config.architectures[0] if hasattr(self.model.config, 'architectures') and len(self.model.config.architectures) > 0 else ""
            target_modules = get_target_modules(model_arc)
            if target_modules is not None:
                print(f"Using target modules for {model_arc}: {target_modules}")
                lora_config = LoraConfig(
                    r=32,
                    lora_alpha=64,
                    target_modules=target_modules,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                
                self.model = get_peft_model(self.model, lora_config)
                self.model = ConversationModelWrapper(self.original_config, base_model=self.model)
                
                self.model.config.use_cache = False
                
                self.model.train()
                self.model.gradient_checkpointing_enable()
                
                print("Successfully created conversation model with LoRA configuration")
                trainable_params = 0
                all_param = 0
                for _, param in self.model.named_parameters():
                    all_param += param.numel()
                    if param.requires_grad:
                        trainable_params += param.numel()
                print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_param:.2f}%)")
                print(f"All params: {all_param:,}")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_repo_path,
                    device_map="auto",
                    trust_remote_code=True,
                    quantization_config=self.quantization_config,
                    torch_dtype=self.dtype
                )
                self.model = ConversationModelWrapper(self.original_config, base_model=self.model)
                
                self.model.config.use_cache = False
                
                self.model.train()
                self.model.gradient_checkpointing_enable()
                
                print("Successfully created conversation model without LoRA configuration")
                trainable_params = 0
                all_param = 0
                for _, param in self.model.named_parameters():
                    all_param += param.numel()
                    if param.requires_grad:
                        trainable_params += param.numel()
                print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_param:.2f}%)")
                print(f"All params: {all_param:,}")
        
            
        except Exception as e:
            print(f"Error creating conversation model: {str(e)}")
            raise
    
    def add_vision(self):
        self.model = prepare_model_for_kbit_training(self.model)
        
        model_type = self.model.config.model_type.lower() if hasattr(self.model.config, 'model_type') else ""
        model_arc = self.model.config.architectures[0] if hasattr(self.model.config, 'architectures') and len(self.model.config.architectures) > 0 else ""
        target_modules = get_target_modules(model_arc)
        if target_modules is not None:
            print(f"Using target modules for {model_arc}: {target_modules}")
            
            lora_config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            config = VisionConfig()
            self.vismodel = VisionModelWrapper(config, lang_model=self.model, model_config=self.model_config)
            
            self.vismodel.train()
            self.vismodel.gradient_checkpointing_enable()
            
            print("Successfully created vision model with LoRA configuration")
            trainable_params = 0
            all_param = 0
            for _, param in self.vismodel.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_param:.2f}%)")
            print(f"All params: {all_param:,}")
        else:
            config = VisionConfig()
            self.vismodel = VisionModelWrapper(config, lang_model=self.model, model_config=self.model_config)
            
            self.vismodel.train()
            self.vismodel.gradient_checkpointing_enable()

            print("Successfully created vision model without LoRA configuration")
            trainable_params = 0
            all_param = 0
            for _, param in self.vismodel.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_param:.2f}%)")
            print(f"All params: {all_param:,}")
        
    def save_regular_model(self):
        """Save the model and all its components with optimizations."""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            # Wrapped Model save
            self.model.save_pretrained(
                self.model_path,
                safe_serialization=True
            )
            # Wrapped Model config save
            self.model.config.save_pretrained(
                self.model_path,
                safe_serialization=True
            )
            
            self.tokenizer.save_pretrained(self.model_path)
            # self.original_config.save_pretrained(self.model_path)
            
            print(f"Successfully saved model to {self.model_path}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise

    def save_vision_model(self):
        """Save the model and all its components with optimizations."""
        try:
            lang_model_path = os.path.join(self.model_path, "lang_model")
            vision_adapter_path = os.path.join(self.model_path, "vision_adapter")
            os.makedirs(lang_model_path, exist_ok=True)
            os.makedirs(vision_adapter_path, exist_ok=True)
            
            # Save language model and tokenizer
            self.tokenizer.save_pretrained(lang_model_path)
            self.model.save_pretrained(
                lang_model_path,
                quantization_config=self.quantization_config,
                torch_dtype=self.dtype,
                safe_serialization=True
            )
            
            # Save vision config only
            # self.vismodel.config.save_pretrained(self.model_path)
            self.vismodel.save_pretrained(self.model_path)
            
            # Save vision components separately
            self.vismodel.vision_model.save_pretrained(
                os.path.join(self.model_path, "vision_model"),
                safe_serialization=True
            )
            torch.save(self.vismodel.vision_adapter.state_dict(), os.path.join(vision_adapter_path, "vision_adapter.pt"))
            print(f"Successfully saved model to {self.model_path}")
        except Exception as e:
            print(f"Error: Failed to save model - {str(e)}")


def load_saved_model(model_path, checkpoint=False):
    variable = Variable()
    dtype = variable.DTYPE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """Load a saved model and its processor."""    
    try:
        # Load the config
        config = AutoConfig.from_pretrained(model_path)
        print(f"Loaded config with model type: {config.model_type}")
        print(f"Model architecture: {config.architectures}")
        is_vision_model = (
            hasattr(config, 'model_type') and config.model_type == "vision-model"
        )
        lang_model_path = os.path.join(model_path, "lang_model")
        local_checkpoint_path = model_path
        vision_model_path = os.path.join(model_path, "vision_model")
        vision_adapter_fpath = os.path.join(model_path, "vision_adapter","vision_adapter.pt")
        
        
        if is_vision_model:
            
            config = VisionConfig()
            # Get architecture safely
            arch = config.architectures[0] if hasattr(config, 'architectures') and config.architectures else None
            target_modules = get_target_modules(arch) if arch else None
            
            # Load vision model components that is addoned on top of language model
            if target_modules is not None:
            
                # Load base model with proper device mapping and dtype
                pefted_lang_model = AutoModelForCausalLM.from_pretrained(
                    lang_model_path,
                    device_map=device,
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
                pefted_lang_model = PeftModel.from_pretrained(pefted_lang_model, lang_model_path)
                pefted_lang_model = pefted_lang_model.to(device).to(dtype)

                # # Enable training on PEFT model - LoRA adapters are frozen by default after loading
                # pefted_lang_model.train()
                
                # # Explicitly enable gradients on LoRA parameters
                # for name, param in pefted_lang_model.named_parameters():
                #     if 'lora' in name.lower():
                #         param.requires_grad = True

                model = VisionModelWrapper(config, lang_model=pefted_lang_model, model_config=ModelConfig())

                # Restore vision adapter if it was saved
                if os.path.exists(vision_adapter_fpath):
                    model.vision_adapter.load_state_dict(torch.load(vision_adapter_fpath, map_location=device))

                tokenizer = AutoTokenizer.from_pretrained(lang_model_path)
                model.config.use_cache = False
                
                # Ensure model is in training mode and has gradients
                model.train()

                if hasattr(model, 'vision_adapter'):
                    for param in model.vision_adapter.parameters():
                        param.requires_grad = True
            else:
                #newly created vision model so it does not have lora
                print("Loading vision model without LoRA...")
                lang_model = AutoModelForCausalLM.from_pretrained(
                    lang_model_path,
                    device_map=device,
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
                model = VisionModelWrapper(config, lang_model=lang_model, model_config=ModelConfig())

                # Restore vision adapter if it was saved
                if os.path.exists(vision_adapter_fpath):
                    model.vision_adapter.load_state_dict(torch.load(vision_adapter_fpath, map_location=device))

                tokenizer = AutoTokenizer.from_pretrained(lang_model_path)
                model.config.use_cache = False
                
                # Ensure model is in training mode and has gradients
                model.train()
                if hasattr(model, 'vision_adapter'):
                    for param in model.vision_adapter.parameters():
                        param.requires_grad = True
                
            return model, tokenizer
        else:
            # Get architecture safely
            arch = config.architectures
            target_modules = get_target_modules(arch[0]) if arch else None

            # Load conversation model that addons on top of base model
            if target_modules is not None:
                print("Loading conversation model with LoRA...")
                pefted_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
                pefted_model = PeftModel.from_pretrained(pefted_model, model_path)
                pefted_model = pefted_model.to(device).to(dtype)
                
                # Enable training on PEFT model - LoRA adapters are frozen by default after loading
                pefted_model.train()
                
                # Explicitly enable gradients on LoRA parameters
                for name, param in pefted_model.named_parameters():
                    if 'lora' in name.lower():
                        param.requires_grad = True
                
                model = ConversationModelWrapper(config, base_model=pefted_model)
                model.config.use_cache = False
                
                model.train()
        
                tokenizer = AutoTokenizer.from_pretrained(local_checkpoint_path)
            else:
                #newly created conversation model so it does not have lora
                print("Loading conversation model... without lora")
                # Direct load without wrapper since it's a standalone model
                
                model = ConversationModel.from_pretrained(
                    model_path,
                    device_map=device,
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
                model.train()
                tokenizer = AutoTokenizer.from_pretrained(local_checkpoint_path)
            return model, tokenizer

        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

AutoConfig.register("conversation-model", ConversationConfig)
AutoModelForCausalLM.register(ConversationConfig, ConversationModelWrapper)


AutoConfig.register("vision-model", VisionConfig)
AutoModelForCausalLM.register(VisionConfig, VisionModelWrapper)