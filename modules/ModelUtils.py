"""
ModelUtils.py – Model wrappers, LoRA helpers, and load/save utilities.

Provides:
  - TARGET_MODULES_MAP       : LoRA target module names keyed by model_type
  - get_target_modules()     : look up target modules for a model_type string
  - QuantizationConfig       : dataclass for 4-bit BitsAndBytes settings
  - ModelConfig              : dataclass for general model-creation settings
  - ConversationModelWrapper : PreTrainedModel wrapping a causal language model
  - VisionAdapter            : MLP projecting CLIP features → language model dim
  - VisionModelWrapper       : PreTrainedModel combining vision encoder + LM
  - VisionProcessor          : ProcessorMixin for joint image + text input
  - CreateModel              : Factory – wrap a base model with LoRA and save it
  - load_saved_model()       : Reload a vision or conversation model from disk
"""

import os
import torch
from pathlib import Path
from dataclasses import dataclass

from transformers import (
    AutoTokenizer, AutoConfig, ProcessorMixin, PreTrainedModel,
    CLIPProcessor, AutoModelForCausalLM, BitsAndBytesConfig, CLIPVisionModel,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from modules.variable import Variable
from modules.models.ConversationModel import ConversationConfig, ConversationModel
from modules.models.VisionModel import VisionConfig

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ── LoRA target-module registry ────────────────────────────────────────────────
# Maps model_type string → attention layer names to apply LoRA on.

TARGET_MODULES_MAP = {
    "gpt2":               ["c_attn", "c_proj"],
    "llama":              ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral":            ["q_proj", "k_proj", "v_proj", "o_proj"],
    "opt":                ["q_proj", "k_proj", "v_proj", "out_proj"],
    "bloom":              ["query_key_value", "dense"],
    "t5":                 ["q", "k", "v", "o"],
    "bert":               ["query", "key", "value", "output.dense"],
    "roberta":            ["query", "key", "value", "output.dense"],
    "gpt_neox":           ["query_key_value", "dense"],
    "falcon":             ["query_key_value", "dense"],
    "mpt":                ["Wqkv", "out_proj"],
    "baichuan":           ["W_pack", "o_proj"],
    "chatglm":            ["query_key_value", "dense"],
    "qwen":               ["c_attn", "c_proj"],
    "phi":                ["Wqkv", "out_proj"],
    "gemma":              ["q_proj", "k_proj", "v_proj", "o_proj"],
    "stablelm":           ["q_proj", "k_proj", "v_proj", "o_proj"],
    "conversation-model": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "vision-model":       ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def get_target_modules(model_type):
    """Return LoRA target module names for model_type, or None if unknown."""
    return TARGET_MODULES_MAP.get(model_type)


# ── Configuration dataclasses ──────────────────────────────────────────────────

@dataclass
class QuantizationConfig:
    """Settings for 4-bit BitsAndBytes quantization."""
    load_in_4bit:             bool        = True
    compute_dtype:            torch.dtype = torch.float32
    quant_type:               str         = "fp4"
    use_double_quant:         bool        = False
    llm_int8_threshold:       float       = 0.0
    llm_int8_has_fp16_weight: bool        = False


@dataclass
class ModelConfig:
    """General settings shared by model-creation helpers."""
    clip_processor_name:    str              = "openai/clip-vit-large-patch14"
    use_fast_tokenizer:     bool             = True
    use_cache:              bool             = False
    gradient_checkpointing: bool             = True
    quantization:           QuantizationConfig = None


# ── ConversationModelWrapper ───────────────────────────────────────────────────

class ConversationModelWrapper(PreTrainedModel):
    """
    Thin wrapper around a causal language model (optionally LoRA-adapted).

    Delegates forward/generate to the inner model (self.bmodel) while
    satisfying HuggingFace Trainer requirements such as gradient checkpointing
    and parameter visibility.
    """
    config_class = ConversationConfig
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, config, **kwargs):
        super().__init__(config)
        base_model = kwargs.get('base_model')
        if base_model is None:
            raise ValueError("base_model is required for ConversationModelWrapper")
        self.bmodel = base_model.to(self.device)
        self.config = config

    # ── Core forward / generate ────────────────────────────────────────────────

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.bmodel(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=labels.to(self.device) if labels is not None else None,
            **kwargs,
        )

    def generate(self, *args, **kwargs):
        return self.bmodel.generate(*args, **kwargs)

    # ── Gradient checkpointing ─────────────────────────────────────────────────

    @property
    def is_gradient_checkpointing(self):
        return self._is_gradient_checkpointing

    @is_gradient_checkpointing.setter
    def is_gradient_checkpointing(self, value):
        self._is_gradient_checkpointing = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.bmodel, "gradient_checkpointing_enable"):
            self.bmodel.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
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

    # ── Parameter exposure for HuggingFace Trainer ────────────────────────────

    def named_parameters(self, *args, **kwargs):
        """Expose inner model parameters to the Trainer."""
        return self.bmodel.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        """Expose inner model parameters to the Trainer."""
        return self.bmodel.parameters(*args, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        if self.bmodel is not None:
            self.bmodel.train(mode)
        return self

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_pretrained(self, save_directory, **kwargs):
        """Save the inner model, preserving PEFT config if present."""
        if hasattr(self.bmodel, 'peft_config'):
            self.bmodel.save_pretrained(save_directory, **kwargs)
        else:
            super().save_pretrained(save_directory, **kwargs)


# ── VisionAdapter ──────────────────────────────────────────────────────────────

class VisionAdapter(torch.nn.Module):
    """
    Three-layer MLP that projects CLIP image embeddings into the language
    model's embedding space.

    Dimensions: clip_dim (1024) → 500 → 1024 → lang_embed_dim
    """

    def __init__(self, lang_embed_dim, clip_dim):
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.layer1 = torch.nn.Linear(clip_dim, 500)
        self.layer2 = torch.nn.Linear(500, 1024)
        self.layer3 = torch.nn.Linear(1024, lang_embed_dim)

    def forward(self, x):
        # Cast to match layer weight dtype (e.g. bfloat16 during training)
        x = x.to(self.layer1.weight.dtype)
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        return x


# ── VisionModelWrapper ─────────────────────────────────────────────────────────

class VisionModelWrapper(PreTrainedModel):
    """
    Combines a frozen CLIP vision encoder with a causal language model.

    Image tokens are projected via VisionAdapter and prepended to the text
    embeddings before each forward/generate pass.
    """
    config_class     = VisionConfig
    NUM_IMAGE_TOKENS = 257  # CLIP outputs 257 tokens (1 CLS + 16×16 patches)

    def __init__(self, config, lang_model=None, model_config=None):
        super().__init__(config)
        self.model_config = model_config or ModelConfig()
        self.config       = config

        # Frozen CLIP vision encoder
        self.vision_model = CLIPVisionModel.from_pretrained(
            self.model_config.clip_processor_name
        )
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # Trainable adapter projecting vision → language embedding dim
        self.vision_adapter = VisionAdapter(1024, 1024)

        self.lang_model                   = lang_model
        self.supports_gradient_checkpointing = True
        self._is_gradient_checkpointing   = False

        # Co-locate vision components with the language model
        if self.lang_model is not None:
            device = next(self.lang_model.parameters()).device
            self.vision_model   = self.vision_model.to(device)
            self.vision_adapter = self.vision_adapter.to(device)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                attend_to_img_tokens=True, labels=None, **kwargs):
        """Build combined vision+text embeddings, then run the language model."""
        embeddings, attention_mask = self._build_embeddings(
            input_ids, attention_mask, pixel_values, attend_to_img_tokens
        )

        # Pad labels to cover the prepended image token positions
        if labels is not None:
            labels = self._pad_labels_for_image(labels, embeddings)

        outputs = self.lang_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        # Compute loss manually if the language model did not return one
        if labels is not None and not hasattr(outputs, 'loss'):
            loss = self._compute_loss(outputs.last_hidden_state, labels)
            outputs = CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=outputs.last_hidden_state,
                past_key_values=getattr(outputs, 'past_key_values', None),
                hidden_states=getattr(outputs, 'hidden_states', None),
                attentions=getattr(outputs, 'attentions', None),
                cross_attentions=getattr(outputs, 'cross_attentions', None),
            )

        return outputs

    # ── Embedding helpers ──────────────────────────────────────────────────────

    def _build_embeddings(self, input_ids, attention_mask, pixel_values, attend_to_img_tokens):
        """
        Embed input_ids and optionally prepend projected image tokens.

        Returns (embeddings, attention_mask) on the language model's device.
        """
        device = next(self.lang_model.parameters()).device
        dtype  = next(self.lang_model.parameters()).dtype

        # Move text inputs to the correct device
        if input_ids      is not None: input_ids      = input_ids.to(device)
        if attention_mask is not None: attention_mask = attention_mask.to(device)

        # Get text token embeddings
        embeddings     = self.lang_model.get_input_embeddings()(input_ids).to(device).to(dtype)
        attention_mask = attention_mask.to(device).to(dtype)

        if pixel_values is not None:
            pixel_values = pixel_values.to(device).to(dtype)

            # Ensure vision components are on the same device
            self.vision_model   = self.vision_model.to(device)
            self.vision_adapter = self.vision_adapter.to(device)

            # Frozen CLIP forward (no gradients), then trainable adapter
            with torch.no_grad():
                image_embeddings = self.vision_model(pixel_values).last_hidden_state.to(dtype).to(device)
            adapted_embeddings = self.vision_adapter(image_embeddings)

            # Prepend image tokens before text tokens
            embeddings     = torch.cat((adapted_embeddings, embeddings), dim=1)
            attention_mask = self._extend_attention_mask(attention_mask, attend_to_img_tokens)

        return embeddings, attention_mask

    def _pad_labels_for_image(self, labels, embeddings):
        """
        Prepend -100 (ignore-index) tokens to labels to align with the
        image token prefix that was added to the embeddings.
        """
        # prefix = number of image tokens prepended
        prefix = embeddings.shape[1] - labels.shape[1]
        if prefix <= 0:
            return labels
        return torch.cat([
            torch.full(
                (labels.shape[0], prefix), -100,
                dtype=labels.dtype, device=labels.device
            ),
            labels,
        ], dim=1)

    def _compute_loss(self, logits, labels):
        """
        Cross-entropy loss with label smoothing.
        Sanitises NaN/Inf values and falls back to zero loss on shape errors.
        """
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size   = shift_logits.size(-1)
        loss_fct     = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        try:
            loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            if torch.isnan(loss) or torch.isinf(loss):
                # Clamp labels to valid range and retry
                loss = loss_fct(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1).clamp(min=0, max=vocab_size - 1),
                )
        except Exception as e:
            print(f"Loss calculation error ({e}) – using zero loss fallback")
            loss = shift_logits.mean() * 0.0

        return loss

    # ── Attention mask extension ────────────────────────────────────────────────

    def _extend_attention_mask(self, atten_mask, attend_to_img=True):
        """
        Extend attention_mask to cover the prepended image tokens.

        attend_to_img=True  → ones  (attend to image tokens)
        attend_to_img=False → zeros (mask out image tokens)
        """
        batch_size, seq_length = atten_mask.shape
        fill_fn = torch.ones if attend_to_img else torch.zeros
        mask = fill_fn(
            (batch_size, seq_length + self.NUM_IMAGE_TOKENS),
            dtype=atten_mask.dtype,
            device=atten_mask.device,
        )
        # Place original text mask at the end (image tokens are prepended)
        mask[:, -seq_length:] = atten_mask
        return mask

    # ── Generate ───────────────────────────────────────────────────────────────

    def generate(self, input_ids=None, attention_mask=None, pixel_values=None,
                 attend_to_img_tokens=True, **kwargs):
        # Support callers that pass these via kwargs as well
        input_ids      = kwargs.pop("input_ids",      input_ids)
        attention_mask = kwargs.pop("attention_mask", attention_mask)
        pixel_values   = kwargs.pop("pixel_values",   pixel_values)

        embeddings, attention_mask = self._build_embeddings(
            input_ids, attention_mask, pixel_values, attend_to_img_tokens
        )

        kwargs.setdefault("max_new_tokens", 100)
        kwargs.setdefault("min_length",     1)
        kwargs.setdefault("num_beams",      4)
        kwargs.setdefault("temperature",    0.7)
        kwargs.setdefault("do_sample",      True)

        return self.lang_model.generate(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )

    # ── Gradient checkpointing ─────────────────────────────────────────────────

    @property
    def is_gradient_checkpointing(self):
        return self._is_gradient_checkpointing

    @is_gradient_checkpointing.setter
    def is_gradient_checkpointing(self, value):
        self._is_gradient_checkpointing = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.vision_model, "gradient_checkpointing_enable"):
            self.vision_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
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


# ── VisionProcessor ────────────────────────────────────────────────────────────

class VisionProcessor(ProcessorMixin):
    """
    Combined image + text processor.

    Wraps a CLIP image processor and a text tokenizer into a single callable
    that returns a dict ready for VisionModelWrapper.
    """
    attributes = ['image_processor', 'tokenizer']

    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer       = tokenizer
        # Use EOS as padding token (standard for causal LMs)
        self.tokenizer.pad_token    = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.chat_template          = self.tokenizer.chat_template

    def __call__(self, text=None, images=None, create_labels=True):
        """
        Process images and/or text into a model-ready dict.

        text          – string or list of strings to tokenize
        images        – PIL Image or list of PIL Images
        create_labels – if True, prepend -100 tokens for image slots in labels
        """
        result = {}

        if images is not None:
            if not isinstance(images, list):
                images = [images]
            result["pixel_values"] = self.image_processor(
                images=images, return_tensors="pt"
            )["pixel_values"]

        if text is not None:
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=250,
                return_tensors="pt",
            )
            result["input_ids"]      = encoded["input_ids"]
            result["attention_mask"] = encoded["attention_mask"]

        if create_labels and "input_ids" in result:
            result = self._add_labels(result)

        return result

    def _add_labels(self, inputs):
        """Prepend NUM_IMAGE_TOKENS=-100 positions so image tokens are ignored in loss."""
        num_image_tokens = VisionModelWrapper.NUM_IMAGE_TOKENS
        inputs['labels'] = torch.cat([
            torch.full(
                (inputs['input_ids'].size(0), num_image_tokens),
                -100,
                dtype=inputs['input_ids'].dtype,
                device=inputs['input_ids'].device,
            ),
            inputs['input_ids'],
        ], dim=1)
        return inputs


# ── CreateModel ────────────────────────────────────────────────────────────────

class CreateModel:
    """
    Wraps an existing base model with LoRA adapters and a custom config,
    then saves the result as a conversation or vision custom model.

    Usage (conversation):
        creator = CreateModel("path/to/base_model", "conversation-model")
        creator.add_conversation()
        creator.save_regular_model()

    Usage (vision):
        creator = CreateModel("path/to/base_model", "vision-model")
        creator.add_vision()
        creator.save_vision_model()
    """

    def __init__(self, model_repo_path, model_category, model_config=None):
        self.model_config    = model_config or ModelConfig()
        self.model_repo_path = Path(model_repo_path)
        self.model_category  = model_category
        self.variable        = Variable()
        self.dtype           = self.variable.DTYPE

        # Build save path: <project_root>/custom_models/<category>/<org>/<name>
        if len(self.model_repo_path.parts) >= 2:
            self.save_name = os.path.join(
                self.model_repo_path.parts[-2],
                self.model_repo_path.parts[-1],
            )
        else:
            self.save_name = self.model_repo_path.name

        self.model_path = (
            Path(__file__).parent.parent.absolute()
            / "custom_models"
            / self.model_category
            / self.save_name
        )
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Build BitsAndBytes quantization config from the dataclass
        quant_cfg = self.model_config.quantization or QuantizationConfig()
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg.load_in_4bit,
            bnb_4bit_compute_dtype=quant_cfg.compute_dtype,
            bnb_4bit_quant_type=quant_cfg.quant_type,
            bnb_4bit_use_double_quant=quant_cfg.use_double_quant,
            llm_int8_threshold=quant_cfg.llm_int8_threshold,
            llm_int8_has_fp16_weight=quant_cfg.llm_int8_has_fp16_weight,
        )

        self.model = self._load_base_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_repo_path,
            use_fast=self.model_config.use_fast_tokenizer,
            trust_remote_code=True,
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            self.model_config.clip_processor_name,
            use_fast=self.model_config.use_fast_tokenizer,
        )
        self.vision_processor = VisionProcessor(self.clip_processor, self.tokenizer)
        self.original_config  = ConversationConfig()
        self.vision_config    = VisionConfig()

    # ── Base model loading ─────────────────────────────────────────────────────

    def _load_base_model(self):
        """Load the base model with quantization, falling back without it on error."""
        print("Loading base model with quantization...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_repo_path,
                quantization_config=self.quantization_config,
                device_map="auto",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            print("Base model loaded with quantization.")
            return model
        except Exception as e:
            print(f"Quantized load failed ({e}), retrying without quantization...")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_repo_path,
                device_map="auto",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            print("Base model loaded without quantization.")
            return model
        except Exception as e2:
            print(f"Model load failed: {e2}")
            return None

    # ── LoRA helpers ───────────────────────────────────────────────────────────

    def _make_lora_config(self, typed):
        """Build a LoraConfig for the given model_type string, or None if unknown."""
        target_modules = get_target_modules(typed)
        if target_modules is None:
            return None
        return LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def _apply_lora(self, model, typed):
        """
        Wrap model with LoRA if target modules are known for typed.
        Returns (peft_model, had_lora: bool).
        """
        lora_cfg = self._make_lora_config(typed)
        if lora_cfg is not None:
            print(f"Applying LoRA for model_type={typed}: {lora_cfg.target_modules}")
            model = get_peft_model(model, lora_cfg)
            return model, True
        print(f"No LoRA target modules for model_type={typed}, skipping LoRA.")
        return model, False

    def _log_trainable(self, model):
        """Print trainable vs total parameter counts."""
        params    = list(model.named_parameters())
        trainable = sum(p.numel() for _, p in params if p.requires_grad)
        total     = sum(p.numel() for _, p in params)
        pct       = 100 * trainable / total if total else 0
        print(f"Trainable: {trainable:,}  Total: {total:,}  ({pct:.2f}%)")

    # ── Public model-creation methods ──────────────────────────────────────────

    def add_conversation(self):
        """Wrap the base model with LoRA + ConversationModelWrapper."""
        if self.model is None:
            print("Error: no base model loaded.")
            return
        try:
            self.model = prepare_model_for_kbit_training(self.model)
            typed      = getattr(self.model.config, 'model_type', '').lower()
            self.model, _ = self._apply_lora(self.model, typed)

            self.model = ConversationModelWrapper(self.original_config, base_model=self.model)
            self.model.config.use_cache = False
            self.model.train()
            self.model.gradient_checkpointing_enable()
            self._log_trainable(self.model)
            print("Conversation model ready.")
        except Exception as e:
            print(f"add_conversation error: {e}")
            raise

    def add_vision(self):
        """Wrap the base model with LoRA + VisionModelWrapper."""
        if self.model is None:
            print("Error: no base model loaded.")
            return
        try:
            self.model = prepare_model_for_kbit_training(self.model)
            typed      = getattr(self.model.config, 'model_type', '').lower()
            self.model, _ = self._apply_lora(self.model, typed)

            self.vismodel = VisionModelWrapper(
                self.vision_config,
                lang_model=self.model,
                model_config=self.model_config,
            )
            self.vismodel.train()
            self.vismodel.gradient_checkpointing_enable()
            self._log_trainable(self.vismodel)
            print("Vision model ready.")
        except Exception as e:
            print(f"add_vision error: {e}")
            raise

    # ── Save methods ───────────────────────────────────────────────────────────

    def save_regular_model(self):
        """Save the conversation model, its config, and the tokenizer."""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            self.model.save_pretrained(self.model_path, safe_serialization=True)
            self.model.config.save_pretrained(self.model_path, safe_serialization=True)
            self.tokenizer.save_pretrained(self.model_path)
            print(f"Saved conversation model to {self.model_path}")
        except Exception as e:
            print(f"save_regular_model error: {e}")
            raise

    def save_vision_model(self):
        """Save vision model components: lang_model, vision_model, and vision_adapter."""
        try:
            lang_path    = self.model_path / "lang_model"
            vis_path     = self.model_path / "vision_model"
            adapter_path = self.model_path / "vision_adapter"
            lang_path.mkdir(parents=True, exist_ok=True)
            vis_path.mkdir(parents=True, exist_ok=True)
            adapter_path.mkdir(parents=True, exist_ok=True)

            # Language model + tokenizer
            self.tokenizer.save_pretrained(str(lang_path))
            self.model.save_pretrained(
                str(lang_path),
                quantization_config=self.quantization_config,
                torch_dtype=self.dtype,
                safe_serialization=True,
            )

            # Full vision wrapper config
            self.vismodel.save_pretrained(str(self.model_path))

            # CLIP vision encoder weights
            self.vismodel.vision_model.save_pretrained(
                str(vis_path), safe_serialization=True
            )

            # Vision adapter weights (plain state dict)
            torch.save(
                self.vismodel.vision_adapter.state_dict(),
                str(adapter_path / "vision_adapter.pt"),
            )
            print(f"Saved vision model to {self.model_path}")
        except Exception as e:
            print(f"save_vision_model error: {e}")


# ── load_saved_model ───────────────────────────────────────────────────────────

def load_saved_model(model_path):
    """
    Load a saved model and its tokenizer from model_path.

    Reads the stored config to decide between vision and conversation loading,
    then dispatches to the appropriate private loader function.
    Returns (model, tokenizer).
    """
    variable = Variable()
    dtype    = variable.DTYPE
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        config    = AutoConfig.from_pretrained(model_path)
        print(f"Config model_type={config.model_type}  architectures={config.architectures}")
        is_vision = getattr(config, 'model_type', '') == "vision-model"

        if is_vision:
            return _load_vision_model(model_path, dtype, device)
        else:
            return _load_conversation_model(model_path, config, dtype, device)

    except Exception as e:
        print(f"load_saved_model error: {e}")
        raise


def _load_vision_model(model_path, dtype, device):
    """
    Load a VisionModelWrapper from a saved directory.

    Expects sub-directories:
      lang_model/                – saved language model (+ optional LoRA adapters)
      vision_adapter/vision_adapter.pt – saved adapter weights
    """
    model_path          = Path(model_path)
    lang_model_path     = model_path / "lang_model"
    vision_adapter_path = model_path / "vision_adapter" / "vision_adapter.pt"
    config              = VisionConfig()

    # Detect LoRA by presence of adapter_config.json in the language model directory
    has_lora = (lang_model_path / "adapter_config.json").exists()

    if has_lora:
        lang_model = AutoModelForCausalLM.from_pretrained(
            str(lang_model_path), device_map=device, torch_dtype=dtype, trust_remote_code=True
        )
        lang_model = PeftModel.from_pretrained(lang_model, str(lang_model_path))
        lang_model = lang_model.to(device).to(dtype)
    else:
        print("Loading vision lang_model without LoRA.")
        lang_model = AutoModelForCausalLM.from_pretrained(
            str(lang_model_path), device_map=device, torch_dtype=dtype, trust_remote_code=True
        )

    model = VisionModelWrapper(config, lang_model=lang_model, model_config=ModelConfig())

    # Restore vision adapter weights if available
    if vision_adapter_path.exists():
        model.vision_adapter.load_state_dict(
            torch.load(str(vision_adapter_path), map_location=device)
        )

    tokenizer              = AutoTokenizer.from_pretrained(str(lang_model_path))
    model.config.use_cache = False
    model.train()

    # Keep adapter trainable
    for param in model.vision_adapter.parameters():
        param.requires_grad = True

    return model, tokenizer


def _load_conversation_model(model_path, config, dtype, device):
    """
    Load a ConversationModelWrapper (or plain ConversationModel) from model_path.

    Detects LoRA by presence of adapter_config.json; falls back to
    ConversationModel.from_pretrained when no adapter is found.
    """
    model_path = Path(model_path)

    # Detect LoRA by presence of adapter_config.json
    has_lora = (model_path / "adapter_config.json").exists()

    if has_lora:
        print(f"Loading conversation model with LoRA from {model_path}")
        peft_model = AutoModelForCausalLM.from_pretrained(
            str(model_path), device_map=device, torch_dtype=dtype, trust_remote_code=True
        )
        peft_model = PeftModel.from_pretrained(peft_model, str(model_path))
        peft_model = peft_model.to(device).to(dtype)
        peft_model.train()

        # Re-enable gradients on LoRA adapter weights
        for name, param in peft_model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True

        model = ConversationModelWrapper(config, base_model=peft_model)
        model.config.use_cache = False
        model.train()
    else:
        print("Loading conversation model without LoRA.")
        model = ConversationModel.from_pretrained(
            str(model_path), device_map=device, torch_dtype=dtype, trust_remote_code=True
        )
        model.train()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    return model, tokenizer


# ── HuggingFace AutoModel registration ────────────────────────────────────────
# Register custom configs so AutoConfig / AutoModelForCausalLM can load them.

AutoConfig.register("conversation-model", ConversationConfig)
AutoModelForCausalLM.register(ConversationConfig, ConversationModelWrapper)

AutoConfig.register("vision-model", VisionConfig)
AutoModelForCausalLM.register(VisionConfig, VisionModelWrapper)
