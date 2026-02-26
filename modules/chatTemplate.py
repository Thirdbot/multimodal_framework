"""
chatTemplate.py – Chat template formatting and dataset preprocessing.

Converts raw HuggingFace datasets (conversations, Q&A pairs, multimodal)
into a unified format that can be tokenized for fine-tuning.
"""

import os
import re
import zipfile
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from colorama import Fore, Style, init
from datasets import Dataset, DatasetDict
from jinja2 import Environment
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoImageProcessor
from transformers.image_utils import load_image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from modules.variable import Variable

init(autoreset=True)

# Environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())
os.environ["TOKENIZERS_PARALLELISM"] = "false"


vars = Variable()

# ── Default model IDs used internally ─────────────────────────────────────────
SENTENCE_TOKENIZER   = "sentence-transformers/msmarco-distilbert-cos-v5"
SENTENCE_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_PROCESSOR_NAME = "microsoft/resnet-50"
CHAT_TEMPLATE_FILE   = vars.CHAT_TEMPLATE_FILE

# Batch size for embedding generation
BATCH_SIZE      = 100
MAX_TEXT_LEN    = 1000
MAX_MULTIMODAL  = 10000

# ── Column-detection patterns (regex) ─────────────────────────────────────────
CONVERSATION_PATTERN = r"^conversations?$"
MESSAGE_PATTERNS     = [r"^messages?$", r"^texts?$", r"^content$"]
MULTIMODAL_PATTERNS  = [r"^image(?:s)?$", r"^audio(?:s)?$", r"^video(?:s)?$"]

# Patterns for irregular datasets (user col, answer col, …)
POTENTIAL_COL_PATTERNS = [
    (
        r"^(?:question|instruction|user|input|Questions?)$",
        r"^(?:answer|response|assistant|output|Answers?)$",
        r"^(?:definition|instruction)$",
        r"^(?:chosen)$",
        r"^(?:rejected)$",
        r"^(?:role)$",
        r"^(?:text)$",
        r"^(?:caption)$",
        r"^(?:label)$",
    )
]

# Mapping from column names to chat roles
ROLE_PATTERNS = {
    "system":    [r"system", r"instruction"],
    "user":      [r"user",   r"human",  r"input"],
    "assistant": [r"assistant", r"gpt", r"output", r"response"],
}


class ChatTemplate:
    """
    Prepare and format datasets for chat-style fine-tuning.

    Steps:
      1. Detect dataset format (conversation / Q&A / multimodal / irregular)
      2. Normalise to a list of {"role": …, "content": …} message dicts
      3. Optionally tokenize using the provided tokenizer
    """

    def __init__(self, tokenizer=None, model_name=None):
        self.tokenizer  = tokenizer
        self.model_name = model_name
        self.variable   = Variable()

        # Select compute device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ChatTemplate device: {self.device}")

        # ── Vision models ──────────────────────────────────────────────
        # ResNet50 backbone – final FC layer removed to get feature embeddings
        self.img_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(self.device)
        self.img_model = torch.nn.Sequential(*list(self.img_model.children())[:-1])
        self.img_model.eval()

        # Standard ImageNet preprocessing pipeline
        self.img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.image_processor = AutoImageProcessor.from_pretrained(IMAGE_PROCESSOR_NAME)

        # ── Text embedding model ───────────────────────────────────────
        self.sentence_tokenizer = AutoTokenizer.from_pretrained(SENTENCE_TOKENIZER)
        self.sentence_model     = SentenceTransformer(SENTENCE_MODEL).to(self.device)
        self.sentence_model.eval()

        # ── Chat template (Jinja) ──────────────────────────────────────
        template_path        = self.variable.chat_template_path
        self.template        = self._compile_template(template_path)

        # Apply the template string to the tokenizer so it formats properly
        tokenizer.chat_template = self._load_template_str(template_path)

    # ── Template loading ───────────────────────────────────────────────────────

    def _load_template_str(self, template_path):
        """Read the .jinja file and inject image-handling logic."""
        raw = self._read_template_file(template_path)
        return self._inject_image_logic(raw)

    def _read_template_file(self, template_path):
        """Return the raw Jinja template string from disk."""
        template_file = template_path / CHAT_TEMPLATE_FILE
        with open(template_file, "r", encoding="utf-8") as f:
            return f.read()

    def _inject_image_logic(self, template_str):
        """
        Patch the template to render <images>…</images> tags when
        message.images is defined.
        """
        image_block = """
                        {% if message.images is defined and message.images %}
                            {% if message.images is string %}
                                <images>{{ message.images }}</images>
                            {% else %}
                                {% for image in message.images %}
                                    <images>{{ image }}</images>
                                {% endfor %}
                            {% endif %}
                        {% endif %}
                        """
        if image_block.strip() not in template_str:
            template_str = template_str.replace(
                "{{ message.content }}",
                "{{ message.content }}" + image_block,
            )
        return template_str

    def _compile_template(self, template_path):
        """Compile the Jinja template object."""
        template_str = self._inject_image_logic(self._read_template_file(template_path))
        return Environment().from_string(template_str)

    # ── Public API ─────────────────────────────────────────────────────────────

    def prepare_dataset(self, dataset_name, dataset, max_length=MAX_TEXT_LEN, Tokenizing=False):
        """
        Main entry point for formatting a dataset.

        dataset_name – string key used for logging and local file lookup
        dataset      – HuggingFace Dataset or DatasetDict
        max_length   – max token length when Tokenizing=True
        Tokenizing   – if True, return tokenized tensors; else return text rows
        """
        # Detect multimodal columns (image / audio / video)
        available_cols = list(dataset.features.keys())
        mul_fields = self._find_multimodal_cols(available_cols)

        if Tokenizing:
            print("Preparing tokenized (embedding) dataset…")
        else:
            print("Preparing template-formatted dataset…")

        try:
            return self._process_dataset(
                dataset_name=dataset_name,
                dataset=dataset,
                mul_field=mul_fields,
                Tokenizing=Tokenizing,
            )
        except Exception as e:
            print(f"{Fore.RED}prepare_dataset error: {e}{Style.RESET_ALL}")
            raise

    def format_message(self, message):
        """Render a list of message dicts to a single prompt string."""
        return self.template.render(messages=message)

    # ── Dataset detection & routing ────────────────────────────────────────────

    def _find_multimodal_cols(self, column_names):
        """Return column names that match image / audio / video patterns."""
        found = []
        for pattern in MULTIMODAL_PATTERNS:
            found.extend(k for k in column_names if re.search(pattern, k, re.IGNORECASE))
        return found

    def _process_dataset(self, dataset_name, dataset, mul_field=None, is_conversation=False,
                         is_checked=False, is_regular=True, Tokenizing=False):
        """
        Recursive dataset router.

        Pass 1 – detect whether it has a 'conversations' column
        Pass 2 – handle conversation datasets
        Pass 3 – handle regular (messages / text) datasets
        Pass 4 – handle irregular (Q&A columns) datasets
        """
        if mul_field is None:
            mul_field = []

        keys = list(dataset.features.keys())

        # Pass 1: first inspection
        if not is_checked and not is_conversation:
            print("Pass 1: checking dataset format…")
            if any(re.search(CONVERSATION_PATTERN, k, re.IGNORECASE) for k in keys):
                is_conversation = True
            return self._process_dataset(
                dataset_name, dataset, mul_field,
                is_conversation=is_conversation, is_checked=True,
                Tokenizing=Tokenizing,
            )

        # Pass 2: known conversation format
        if is_checked and is_conversation:
            print("Pass 2: conversation dataset")
            return self._handle_separated(dataset_name, dataset, "conversations", mul_field, Tokenizing)

        # Pass 3: regular message / text columns
        if is_checked and not is_conversation and is_regular:
            print("Pass 3: looking for message / text columns")
            for pattern in MESSAGE_PATTERNS:
                matching = [k for k in keys if re.search(pattern, k, re.IGNORECASE)]
                if matching:
                    col = matching[0]
                    if isinstance(dataset[col], list):
                        return self._handle_separated(dataset_name, dataset, col, mul_field, Tokenizing)
            # No standard column found – fall through to irregular handling
            return self._process_dataset(
                dataset_name, dataset, mul_field,
                is_conversation=False, is_checked=True, is_regular=False,
                Tokenizing=Tokenizing,
            )

        # Pass 4: irregular – Q&A / role-text pairs
        if is_checked and not is_conversation and not is_regular:
            print("Pass 4: irregular dataset")
            return self._handle_irregular(dataset_name, dataset, mul_field, Tokenizing)

        print("Warning: no processing path found")
        return None

    # ── Format handlers ────────────────────────────────────────────────────────

    def _handle_separated(self, dataset_name, dataset, key, mul_field, Tokenizing):
        """
        Handle datasets that already have a list-of-messages column.

        Tokenizing=False → return the processed (template-formatted) dataset
        Tokenizing=True  → return a tokenized DatasetDict ready for Trainer
        """
        if not Tokenizing:
            return dataset

        if not mul_field:
            return self._tokenize_text_only(dataset, key)
        else:
            return self._tokenize_multimodal(dataset, key, mul_field, dataset_name)

    def _handle_irregular(self, dataset_name, dataset, mul_field, Tokenizing):
        """
        Convert irregular (Q&A / role-text / single-text) columns into
        a conversations list, then re-route through _process_dataset.
        """
        keys = list(dataset.features.keys())
        conversations = []

        for patterns in POTENTIAL_COL_PATTERNS:
            p_user, p_asst, p_sys, p_chosen, p_rejected, p_role, p_text, p_caption, p_label = patterns

            col_user    = self._match_col(p_user,     keys)
            col_asst    = self._match_col(p_asst,     keys)
            col_sys     = self._match_col(p_sys,      keys)
            col_chosen  = self._match_col(p_chosen,   keys)
            col_rejected= self._match_col(p_rejected, keys)
            col_role    = self._match_col(p_role,     keys)
            col_text    = self._match_col(p_text,     keys)
            col_caption = self._match_col(p_caption,  keys)
            col_label   = self._match_col(p_label,    keys)

            if col_user and col_asst and col_sys:
                # instruction + Q&A format
                for user, asst, sys in zip(dataset[col_user], dataset[col_asst], dataset[col_sys]):
                    conversations.append([
                        {"role": "system",    "content": sys},
                        {"role": "user",      "content": user},
                        {"role": "assistant", "content": asst},
                    ])
            elif col_user and col_asst:
                # simple Q&A format
                for user, asst in zip(dataset[col_user], dataset[col_asst]):
                    conversations.append([
                        {"role": "user",      "content": user},
                        {"role": "assistant", "content": asst},
                    ])
            elif col_chosen and col_rejected:
                # preference / RLHF format
                for chosen, rejected in zip(dataset[col_chosen], dataset[col_rejected]):
                    conversations.append([
                        {"role": "user",      "content": chosen},
                        {"role": "assistant", "content": rejected},
                    ])
            elif col_role and col_text:
                # role + text columns
                for role, text in zip(dataset[col_role], dataset[col_text]):
                    conversations.append([{"role": role, "content": text}])
            elif col_caption:
                for text in dataset[col_caption]:
                    conversations.append([
                        {"role": "user",      "content": "What is in this image?"},
                        {"role": "assistant", "content": text},
                    ])
            elif col_label:
                for text in dataset[col_label]:
                    conversations.append([
                        {"role": "user",      "content": "What is in this image?"},
                        {"role": "assistant", "content": text},
                    ])
            else:
                print("No matching column pattern found for this dataset.")
                return None

        if not conversations:
            return None

        # Build a normalised dataset and re-route
        built = {"conversations": conversations}
        if mul_field:
            for col in mul_field:
                built[col] = [row for row in dataset[col] if row is not None]

        normalised = Dataset.from_dict(built)
        return self._process_dataset(
            dataset_name, normalised, mul_field,
            is_conversation=False, is_checked=False,
            Tokenizing=Tokenizing,
        )

    # ── Tokenisation helpers ───────────────────────────────────────────────────

    def _tokenize_text_only(self, dataset, key):
        """Tokenize a text-only conversation dataset → DatasetDict."""
        # Render each conversation through the chat template
        texts = []
        for conv in dataset[key]:
            if isinstance(conv, list):
                texts.append(self.format_message(conv))

        if not texts:
            return None

        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )

        # Build labels; mask padding and special tokens
        labels = encodings["input_ids"].clone()
        labels[encodings["special_tokens_mask"] == 1] = -100
        labels[encodings["attention_mask"] == 0]      = -100

        if torch.all(labels == -100):
            print("Warning: all labels masked – check your tokenizer config.")
            return None

        return DatasetDict({
            "train": Dataset.from_dict({
                "input_ids":      encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels":         labels,
            })
        })

    def _tokenize_multimodal(self, dataset, key, mul_fields, dataset_name):
        """Tokenize a multimodal dataset (text + images) → DatasetDict."""
        texts = []
        for conv in dataset[key]:
            if isinstance(conv, (dict, list)):
                texts.append(self.format_message(conv))
        if not texts:
            return None

        text_enc = self.tokenizer(
            texts,
            padding=True, truncation=True,
            max_length=MAX_MULTIMODAL,
            return_tensors="pt",
        )

        result = {
            "input_ids":      text_enc["input_ids"],
            "attention_mask": text_enc["attention_mask"],
        }

        processed_images = []
        valid_indices    = []
        for field in mul_fields:
            for idx, img in enumerate(dataset[field]):
                try:
                    proc = self.image_processor(img, return_tensors="pt")
                    processed_images.append(proc["pixel_values"].squeeze(0))
                    valid_indices.append(idx)
                except Exception:
                    continue

        if processed_images:
            result["input_ids"]      = text_enc["input_ids"][valid_indices]
            result["attention_mask"] = text_enc["attention_mask"][valid_indices]
            result["pixel_values"]   = processed_images

        return DatasetDict({"train": Dataset.from_dict(result)})

    # ── Embedding helpers ──────────────────────────────────────────────────────

    def text_embedding(self, text):
        """
        Compute a normalised sentence embedding for a text string.
        Returns a numpy array or None on error.
        """
        try:
            if isinstance(text, (bytes, bytearray)):
                text = text.decode("utf-8")
            self.sentence_model.eval()
            with torch.no_grad():
                output = self.sentence_model.encode(text)
                return F.normalize(output, p=2, dim=1).detach().cpu().numpy()
        except Exception as e:
            print(f"text_embedding error: {e}")
            return None

    def image_embedding(self, image_obj):
        """
        Compute a normalised image embedding via ResNet50.
        Accepts a PIL Image. Returns a numpy array or None on error.
        """
        try:
            if not isinstance(image_obj, Image.Image):
                print(f"Expected PIL.Image, got {type(image_obj)}")
                return None
            tensor = self.img_transform(image_obj).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.img_model(tensor).squeeze()
                return F.normalize(out.unsqueeze(0), p=2, dim=1).detach().cpu().numpy()
        except Exception as e:
            print(f"image_embedding error: {e}")
            return None

    # ── Message content helpers ────────────────────────────────────────────────

    def _get_role_keys(self, item):
        """
        Detect which key is the 'role' and which is the 'content'
        in a message dict.  Returns (role_key, content_key) or (None, None).
        """
        if isinstance(item, dict):
            keys = tuple(item.keys())
        elif isinstance(item, list) and item:
            keys = tuple(item[0].keys())
        else:
            return None, None

        ROLE_KEYS    = {"role", "from"}
        CONTENT_KEYS = {"content", "value"}

        if keys[0] in ROLE_KEYS:
            return keys[0], keys[1]
        if keys[1] in ROLE_KEYS:
            return keys[1], keys[0]
        if keys[0] in CONTENT_KEYS:
            return keys[1], keys[0]
        if keys[1] in CONTENT_KEYS:
            return keys[0], keys[1]

        return None, None

    def get_text_content(self, role_key, content_key, full_data, Tokenizing=False):
        """
        Walk through all messages in full_data and optionally embed
        the text content. Returns the (possibly modified) list or None.
        """
        try:
            for msg in full_data:
                if not isinstance(msg.get(content_key), str):
                    continue
                if Tokenizing:
                    embedded = self.text_embedding(msg[content_key])
                    if embedded is not None:
                        msg[content_key] = embedded
            return full_data
        except Exception as e:
            print(f"get_text_content error: {e}")
            return None

    # ── File / multimodal helpers ──────────────────────────────────────────────

    def get_mul_file(self, data_name, dataset_name, Tokenizing=False):
        """
        Locate a file by name in the local dataset directory,
        including inside zip archives.

        Returns the processed content (image array, path string) or None.
        """
        short    = dataset_name.split("/")[-1]
        root     = self.variable.WORKSPACE / "repositories" / "datasets" / short

        if not root.exists():
            print(f"Local dataset not found: {root}")
            return None

        files   = [p for p in root.iterdir() if p.is_file()]
        zips    = [p for p in files if p.suffix == ".zip"]
        folders = [p for p in root.iterdir() if p.is_dir()]
        pattern = re.compile(r"^" + re.escape(data_name) + r"$")

        # Search plain files
        for f in tqdm(files, desc="Searching files"):
            if pattern.match(f.name):
                return self._read_file(str(f), zip_ref=None, Tokenizing=Tokenizing)

        # Search inside subdirectories
        for folder in folders:
            for f in tqdm(Path(folder).rglob(data_name), desc="Searching folder"):
                return self._read_file(str(f), zip_ref=None, Tokenizing=Tokenizing)

        # Search inside zip archives
        for zip_path in zips:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in tqdm(zf.namelist(), desc="Searching zip"):
                    if pattern.match(name):
                        return self._read_file(name, zip_ref=zf, Tokenizing=Tokenizing)

        return None

    def _read_file(self, file_path, zip_ref=None, Tokenizing=False):
        """
        Read and optionally process an image file.

        zip_ref – open ZipFile object if the file is inside a zip, else None
        Returns processed pixel values (Tokenizing=True) or the path string.
        """
        ext = str(file_path).split(".")[-1].lower()

        if ext not in ("jpg", "jpeg", "png"):
            # Audio / other types not yet supported
            return None

        try:
            if zip_ref:
                image_path = os.path.join(str(zip_ref.filename), file_path)
                if Tokenizing:
                    return np.array(load_image(image_path))
                return image_path
            else:
                if Tokenizing:
                    img  = load_image(file_path)
                    proc = self.image_processor(img, return_tensors="pt")
                    return proc["pixel_values"]
                return file_path
        except Exception as e:
            print(f"_read_file error [{file_path}]: {e}")
            return None

    # ── Batch / dependency helpers ─────────────────────────────────────────────

    def process_dataset_dependencies(self, dataset_name, dataset, key, mul_field=None, Tokenizing=False):
        """
        Process a dataset in batches and collect embedded messages / images.

        Returns embedded_messages (list) when no mul_field,
        or combined [{"conversations": …, "image": …}] list when mul_field is set.
        """
        total      = len(dataset[key])
        batches    = self._make_batches(dataset, key, mul_field, total)

        if not mul_field:
            return self._collect_text_batches(dataset_name, batches, key, Tokenizing)
        else:
            return self._collect_multimodal_batches(dataset_name, batches, key, mul_field, Tokenizing)

    def _make_batches(self, dataset, key, mul_field, total):
        """Split dataset into BATCH_SIZE chunks."""
        batches = []
        for start in range(0, total, BATCH_SIZE):
            end   = min(start + BATCH_SIZE, total)
            print(f"Batching: {end}/{total}", end="\r")
            batch = {key: dataset[key][start:end]}
            if mul_field:
                for col in mul_field:
                    batch[col] = dataset[col][start:end]
            batches.append(batch)
        print()
        return batches

    def _collect_text_batches(self, dataset_name, batches, key, Tokenizing):
        """Collect embedded messages from text-only batches."""
        messages = []
        for batch in batches:
            role_key, content_key = self._get_role_keys(
                batch[key][0] if batch[key] else {}
            )
            if role_key is None:
                continue
            for item in tqdm(batch[key], desc="Processing text"):
                result = self.get_text_content(role_key, content_key, item, Tokenizing)
                if result is not None:
                    messages.append(result)
        print(f"Completed: {len(messages)} items")
        return messages

    def _collect_multimodal_batches(self, dataset_name, batches, key, mul_field, Tokenizing):
        """Collect embedded messages and images from multimodal batches."""
        all_messages = []
        all_images   = []
        for batch in batches:
            role_key, content_key = self._get_role_keys(
                batch[key][0] if batch[key] else {}
            )
            if role_key is None:
                continue
            for text_item, *mul_items in zip(batch[key], *[batch[col] for col in mul_field]):
                text_result = self.get_text_content(role_key, content_key, text_item, Tokenizing)
                for mul_item in mul_items:
                    img = self.get_mul_file(mul_item, dataset_name, Tokenizing)
                    if text_result and img:
                        all_messages.append(text_result)
                        all_images.append(img)

        if all_messages and all_images:
            return [{"conversations": m, "image": i}
                    for m, i in zip(all_messages, all_images)]
        return []

    # ── Column matching helper ─────────────────────────────────────────────────

    def _match_col(self, pattern, column_names):
        """Return the first column name matching the regex pattern, or None."""
        for col in column_names:
            if re.search(pattern, col, re.IGNORECASE):
                return col
        return None
