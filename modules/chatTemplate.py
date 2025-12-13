import os
import zipfile
import re
import multiprocessing as mp
from dataclasses import dataclass
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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@dataclass
class ChatTemplateConfig:
    """Runtime configuration for ChatTemplate to avoid hard-coding paths and model ids."""

    chat_template_filename: str = "chat_template_conversation.jinja"
    sentence_tokenizer_name: str = "sentence-transformers/msmarco-distilbert-cos-v5"
    sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_processor_name: str = "microsoft/resnet-50"
    image_model_weights: ResNet50_Weights = ResNet50_Weights.IMAGENET1K_V2
    device_override: str | None = None
    max_text_length: int = 1000
    max_multimodal_length: int = 10000
    batch_size: int = 100
    repository_root: Path | None = None

class ChatTemplate:
    """Class for handling chat templates and conversation formatting"""
    
    # Regex patterns for column matching
    CONVERSATION_PATTERN = r'^conversations?$'
    MESSAGE_PATTERNS = [r'^messages?$', r'^texts?$', r'^content$']
    MULTIMODAL_PATTERNS = [r'^image(?:s)?$', r'^audio(?:s)?$', r'^video(?:s)?$']
    POTENTIAL_COLUMNS_PATTERNS = [
        (r'^(?:question|instruction|user|input|Questions?)$', 
         r'^(?:answer|response|assistant|output|Answers?)$',
         r'^(?:definition|instruction)$',
         r'^(?:chosen)$',
         r'^(?:rejected)$',
         r'^(?:role)$',
         r'^(?:text)$',
         r'^(?:caption)$',
         r'^(?:label)$')
    ]
    
    # Role patterns and mappings
    ROLE_PATTERNS = {
        'system': [r'system', r'instruction'],
        'user': [r'user', r'human', r'input'],
        'assistant': [r'assistant', r'gpt', r'output', r'response']
    }
    
    def __init__(self, tokenizer=None, model_name=None, config: ChatTemplateConfig | None = None):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.config = config or ChatTemplateConfig()

        # Resolve repository root from config or default to project root
        self.variable = Variable()
        default_repo_root = Path(__file__).parent.parent.absolute()
        self.repository_root = self.config.repository_root or default_repo_root

        # Device selection is configurable
        device_str = self.config.device_override or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        print(f"Using device: {self.device}")

        # Initialize image model with ResNet for reliable embeddings
        self.img_model = resnet50(weights=self.config.image_model_weights).to(self.device)
        # Remove the final classification layer
        self.img_model = torch.nn.Sequential(*(list(self.img_model.children())[:-1]))
        self.img_model.eval()

        # Initialize image preprocessing
        self.img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.sentence_tokenizer = AutoTokenizer.from_pretrained(self.config.sentence_tokenizer_name)
        self.image_processor = AutoImageProcessor.from_pretrained(self.config.image_processor_name)

        self.sentence_model = SentenceTransformer(self.config.sentence_model_name).to(self.device)
        self.sentence_model.eval()  # Ensure model is in eval mode from the start

        self.chat_template_path = self.variable.chat_template_path
        self.template = self.load_template_from_folder()

        self.tokenizer.chat_template = self.str_template(self.chat_template_path)
    
    def _read_template_str(self, template_path: Path) -> str:
        template_file = template_path / self.config.chat_template_filename
        with open(template_file, "r", encoding="utf-8") as f:
            return f.read()

    def _inject_image_logic(self, template_str: str) -> str:
        image_logic = """
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
        if image_logic.strip() not in template_str:
            template_str = template_str.replace(
                "{{ message.content }}",
                "{{ message.content }}" + image_logic
            )
        return template_str

    def str_template(self, template_path: Path):
        template_str = self._read_template_str(template_path)
        return self._inject_image_logic(template_str)

    def load_template_from_folder(self):
        template_str = self._inject_image_logic(self._read_template_str(self.chat_template_path))
        envi = Environment()
        template = envi.from_string(template_str)
        return template

    def seperated_data(self,dataset_name,dataset,keys,mul_field=[],Tokenizing=False):
        dataset_keys = dataset.features.keys()
        
        if Tokenizing: 
            if len(mul_field) == 0:
                print("no multimodality to tokenize")
                # Format text data for tokenization
                formatted_texts = []
                for conversation in dataset[keys]:
                    if isinstance(conversation, list):
                        formatted_text = self.format_message(conversation)
                        formatted_texts.append(formatted_text)
                
                if not formatted_texts: 
                    return None
                
                # Tokenize all texts for training
                print("Tokenizing texts for training")
                train_encodings = self.tokenizer(
                    formatted_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_text_length,
                    return_tensors="pt",
                    return_attention_mask=True,
                    return_special_tokens_mask=True
                )
                
                # Create labels for language modeling
                labels = train_encodings['input_ids'].clone()
                
                # Create attention mask for non-padded tokens
                attention_mask = train_encodings['attention_mask']
                
                # Set labels to -100 for padding tokens and special tokens
                special_tokens_mask = train_encodings['special_tokens_mask']
                labels[special_tokens_mask == 1] = -100  # Ignore special tokens
                labels[attention_mask == 0] = -100  # Ignore padding tokens
                
                # Verify we have valid labels
                if torch.all(labels == -100):
                    print("Warning: All labels are masked, this will result in zero loss")
                    return None
                
                # Create train dataset
                train_dataset = Dataset.from_dict({
                    'input_ids': train_encodings['input_ids'],
                    'attention_mask': train_encodings['attention_mask'],
                    'labels': labels
                })
                
                # Return DatasetDict with only train split
                return DatasetDict({
                    'train': train_dataset
                })
            
            # Check for multimodal content if needed
            else:
                # Format and tokenize text data
                formatted_texts = []
                for conversation in dataset[keys]:
                    if isinstance(conversation, dict) or isinstance(conversation, list):
                        formatted_text = self.format_message(conversation)
                        formatted_texts.append(formatted_text)
                
                if not formatted_texts:
                    return None
                
                # Tokenize text
                text_tokenized = self.tokenizer(
                    formatted_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_multimodal_length,
                    return_tensors="pt"
                )
                
                # Process multimodal data
                multimodal_data = {
                    'input_ids': text_tokenized['input_ids'],
                    'attention_mask': text_tokenized['attention_mask'],
                }
                
                processed_images,valid_image_idx = [],[]

                for mul in mul_field:
                    set_data = dataset[mul]
                    print(f"processing {mul} columns")
                    if mul in dataset_keys:
                        # Ensure images are in the correct format
                        for idx, img in enumerate(set_data):
                            try:
                                processed = self.image_processor(img, return_tensors="pt")

                                processed_images.append(processed['pixel_values'].squeeze(0))

                                valid_image_idx.append(idx)
                            except Exception as e:
                                continue

                        if len(processed_images) == 0:
                            processed_images = self.process_dataset_dependencies(dataset_name, dataset, keys, mul_field)

                if processed_images:
                    print("Stacking processed images")
                    # Stack all processed images into a single batch
                    # image_batch = torch.stack(torch.tensor(processed_images), dim=0)
                    multimodal_data['input_ids'] = text_tokenized['input_ids'][valid_image_idx]
                    multimodal_data['attention_mask'] = text_tokenized['attention_mask'][valid_image_idx]
                    multimodal_data['pixel_values'] = processed_images
                    # multimodal_data['pixel_values'] = processed_images
                
                # Create and return dataset with only training data
                train_dataset = Dataset.from_dict(multimodal_data)
                
                return DatasetDict({
                    'train': train_dataset
                })
        
        else:
            return dataset


    def process_dataset_dependencies(self, dataset_name, dataset, keys, mul_field=None, Tokenizing=False):
        print(f"Processing embedded dataset")
        total_items = len(dataset[keys])

        batch_size = self.config.batch_size
        batch_list = []

        for index in range(0, total_items, batch_size):
            end_idx = min(index + batch_size, total_items)
            print(f"Processing batch: {end_idx}/{total_items}", end="\r")
            batch = {keys: dataset[keys][index:end_idx]}
            if mul_field and len(mul_field) > 0:
                for mul in mul_field:
                    batch[mul] = dataset[mul][index:end_idx]
            batch_list.append(batch)
        print("\n")

        if not mul_field or len(mul_field) == 0:
            embedded_messages = []
            for batch in batch_list:
                batch_result = self.mul_process(dataset_name, batch, keys, None, Tokenizing)
                if batch_result is not None:
                    messages, _ = batch_result
                    if messages:
                        embedded_messages.extend(messages)
            print(f"\nCompleted message processing {len(embedded_messages)} items")
            return embedded_messages

        else:
            embedded_messages = []
            embedded_images = []
            for batch in batch_list:
                batch_messages, batch_images = self.mul_process(dataset_name, batch, keys, mul_field, Tokenizing)
                if batch_messages:
                    embedded_messages.extend(batch_messages)
                if batch_images:
                    embedded_images.extend(batch_images)
            print(f"\nCompleted mul processing {len(embedded_messages)} items")
            combined_data = []
            for msg, img in zip(embedded_messages, embedded_images):
                combined_data.append({
                    'conversations': msg,
                    'image': img
                })
            return combined_data

    def mul_process(self,dataset_name,dataset,keys,mul_field=None,Tokenizing=False):
        if mul_field is None:
            # Create new lists to store embedded data
            batch_data = []
            
            # Get the data for the specified key
            data = dataset[keys]
                
            if not data:
                print(f"Warning: No data found for key {keys}")
                return None, None  # Return tuple of None values to match expected format
                
            for list_data in tqdm(data, desc="Processing non-multimodal sub-batches"):
                if not isinstance(list_data, (dict, list)):
                    print(f"Warning: Invalid list_data format: {type(list_data)}")
                    continue
                    
                # Handle both dict and list formats
                if isinstance(list_data, dict):
                    get_keys = tuple(list_data.keys())
                elif isinstance(list_data, list) and len(list_data) > 0:
                    get_keys = tuple(list_data[0].keys())
                else:
                    print(f"Warning: Invalid list_data structure")
                    continue
                
                if get_keys[0] in ('role','from'):
                    role = get_keys[0]
                    content = get_keys[1]
                elif get_keys[1] in ('content','value'):
                    role = get_keys[1]
                    content = get_keys[0]
                elif get_keys[0] in ('content','value'):
                    role = get_keys[1]
                    content = get_keys[0]
                elif get_keys[1] in ('role','from'):
                    role = get_keys[1]
                    content = get_keys[0]
                else:
                    print(f"Warning: Invalid keys: {get_keys}")
                    continue
                
                processed_content = self.get_text_content(role, content, list_data,Tokenizing)
                if processed_content is not None:
                    batch_data.append(processed_content)
            
            return batch_data, None  # Return tuple with batch_data and None for images

        elif len(mul_field) > 0:
            # Create new lists to store embedded data
            embedded_messages = []
            embedded_images = []
            
            # Get the data for the specified key
            text_data = dataset[keys]
                
            if not text_data:
                print(f"Warning: No text data found for key {keys}")
                return None, None
            
            # Process each multimodal field
            for mul in mul_field:
                try:
                    # print(f"Processing field: {mul}")
                    
                    # Get the data for this field
                    mul_data = dataset[mul]
                    if not mul_data:
                        # print(f"Warning: No data found for field {mul}")
                        continue
                    
                    # Process each item in the dataset
                    for text_item, mul_item in tqdm(zip(text_data, mul_data), desc=f"Processing {mul} field"):
                        try:
                            # Get the keys for text processing
                            if isinstance(text_item, dict):
                                get_keys = tuple(text_item.keys())
                            elif isinstance(text_item, list) and len(text_item) > 0:
                                get_keys = tuple(text_item[0].keys())
                            else:
                                print(f"Warning: Unexpected text item format: {type(text_item)}")
                                continue
                                
                            if get_keys[0] in ('role','from'):
                                role = get_keys[0]
                                content = get_keys[1]
                            elif get_keys[1] in ('content','value'):
                                role = get_keys[1]
                                content = get_keys[0]
                            elif get_keys[0] in ('content','value'):
                                role = get_keys[1]
                                content = get_keys[0]
                            elif get_keys[1] in ('role','from'):
                                role = get_keys[1]
                                content = get_keys[0]
                            else:
                                print(f"Warning: Invalid keys: {get_keys}")
                                continue
                            
                            # Process the multimodal content
                            processed_messages, processed_images = self.get_mul_content(
                                dataset_name=dataset_name,
                                mul_content=mul_item,
                                full_content=text_item,
                                role=role,
                                content=content,
                                Tokenizing=Tokenizing
                            )
                            
                            if processed_messages:
                                embedded_messages.extend(processed_messages)
                            if processed_images:
                                embedded_images.extend(processed_images)
                                
                        except Exception as e:
                            print(f"Error processing item: {str(e)}")
                            continue
                            
                except Exception as e:
                    print(f"Error processing {mul} field: {str(e)}")
                    continue
            
            return (embedded_messages, embedded_images) if embedded_messages and embedded_images else (None, None)

    def get_text_content(self, role, content, full_data,Tokenizing=False):
        # Process all messages in the conversation
        try:
            for msg in full_data:
                if isinstance(msg[content], str):
                    # Determine role type using regex patterns
                    role_type = None
                    for role_type, patterns in self.ROLE_PATTERNS.items():
                        if any(re.search(pattern, msg[role], re.IGNORECASE) for pattern in patterns):
                            role_type = role_type
                            break
                    if Tokenizing:
                        embedded_content = self.text_embedding(msg[content])
                    else:
                        embedded_content = msg[content]
                        
                    if embedded_content is not None:
                        msg[content] = embedded_content

            return full_data
        except Exception as e:
            print(f"Error processing text content: {str(e)}")
            return None
    #make a dataset of mul content
    def get_mul_content(self,dataset_name,mul_content,full_content,role,content,Tokenizing=False):
        # Convert any supported input into a list and embed each element once
        mul_items = mul_content if isinstance(mul_content, list) else [mul_content]
        embedded_items = []
        for item in mul_items:
            embedded = self.get_mul_file(item, dataset_name, Tokenizing=Tokenizing)
            if embedded is not None:
                embedded_items.append(embedded)

        if not embedded_items:
            return [], []

        processed_messages = []
        processed_content = self.get_text_content(role, content, full_content, Tokenizing=Tokenizing)
        if processed_content is not None:
            processed_messages.append(processed_content)

        return processed_messages, embedded_items
    
    #get file from local repository
    def get_mul_file(self,data_name,dataset_name,Tokenizing=False):
        short_name = dataset_name.split("/")[-1]
        local_dataset_path = self.repository_root / "repositories" / "datasets" / short_name
 
        file_in_path = [path for path in Path(local_dataset_path).iterdir() if path.is_file()]
        zip_file_in_path = [path for path in Path(local_dataset_path).iterdir() if path.is_file() and path.suffix == ".zip"]
        folder_in_path = [path for path in Path(local_dataset_path).iterdir() if path.is_dir()]
        if not local_dataset_path.exists():
            print(f"Dataset {dataset_name} not found in local repository")
            return False
        pattern = r"^" + re.escape(data_name) + r"$"
        
        for file in tqdm(file_in_path, desc="Searching for file"):
            if re.match(pattern, file.name):
                # print("file founded")
                return self.read_file(file.name,zip=None,Tokenizing=Tokenizing)

        for folder in folder_in_path:
            # print("folder founded")
            for file_path in tqdm(Path(folder).rglob(data_name), desc="Searching for file"):
                # print("files founded")
                return self.read_file(file_path,zip=None,Tokenizing=Tokenizing)

        for zip_file in zip_file_in_path:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for zip_file_name in tqdm(zip_ref.namelist(), desc="Searching for file"):
                    if re.match(pattern, zip_file_name):
                        # print("zip file founded")
                        return self.read_file(zip_file_name,zip=zip_file,Tokenizing=Tokenizing)

    #get actual data from file
    def read_file(self,file_path,zip=None,Tokenizing=False):
        file_ext = file_path.split('.')[-1].lower()
        if zip:
            # image
            if file_ext in ['jpg', 'jpeg', 'png']:
                try:
                    # Load and process image from zip
                    image_path = str(os.path.join(zip,file_path))
                    if Tokenizing:
                        image_obj = load_image(image_path)
                        # processed = self.image_processor(image_obj, return_tensors="pt")
                        
                        # return processed['pixel_values']
                        return np.array(image_obj)
                    else:
                        return image_path

                except Exception as e:
                    print(f"Error processing image from zip {file_path}: {str(e)}")
                    return None
            elif file_ext in ['mp3', 'wav']:
                try:
                    get_file_obj = zip.open(file_path)
                    # audio_obj = AudioSegment.from_file(get_file_obj)
                    # return self.audio_embedding(audio_obj)
                    return None
                except Exception as e:
                    print(f"Error processing audio {file_path}: {str(e)}")
                    return None
        else:
            #image
            if file_ext in ['jpg', 'jpeg', 'png']:
                try:
                    if Tokenizing:
                        image_obj = load_image(file_path)
                        processed = self.image_processor(image_obj, return_tensors="pt")
                        return processed['pixel_values']
                    else:
                        return file_path
    
                except Exception as e:
                    print(f"Error processing image {file_path}: {str(e)}")
                return None
            elif file_ext in ['mp3', 'wav']:
                try:
                    get_file_obj = open(file_path, 'rb')
                    # audio_obj = AudioSegment.from_file(get_file_obj)
                    # return self.audio_embedding(audio_obj)
                    return None
                except Exception as e:
                    print(f"Error processing audio {file_path}: {str(e)}")
                    return None
    


    def text_embedding(self,text):
        try:
            if isinstance(text, (bytes, bytearray)):
                text = text.decode('utf-8')
            
            # Ensure model is in eval mode
            self.sentence_model.eval()
            
            # Tokenize the text
            # encoded_input = self.sentence_tokenizer(
            #     str(text), 
            #     padding=True, 
            #     truncation=True, 
            #     max_length=512
            # )
            # print(f"encoded_input: {encoded_input}")
            # Move inputs to the same device as the model
            # encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                # Get model outputs
                outputs = self.sentence_model.encode(text)
                # print(f"outputs: {outputs}")
                
                # Get the last hidden state
                # last_hidden_state = outputs.last_hidden_state
                
                # Get attention mask
                # attention_mask = encoded_input['attention_mask']
                
                # # Mean pooling
                # token_embeddings = last_hidden_state
                # input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                # embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # # Normalize embeddings
                normalized_embeddings = F.normalize(outputs, p=2, dim=1)
                
                # # Convert to numpy
                final_embeddings = normalized_embeddings.detach().cpu().numpy()
                
                
                return final_embeddings
            
        except Exception as e:
            print(f"Error creating text embedding: {str(e)}")
            return None
    
    def image_embedding(self,image_obj):
        try:
            if isinstance(image_obj, Image.Image):
             
                img_tensor = self.img_transform(image_obj).unsqueeze(0).to(self.device)
              
                with torch.no_grad():
                    # Get model outputs
                    outputs = self.img_model(img_tensor)
                    # Remove the batch dimension
                    outputs = outputs.squeeze()
                    

                    normalized_embeddings = F.normalize(outputs.unsqueeze(0), p=2, dim=1)
                    
              
                    final_embeddings = normalized_embeddings.detach().cpu().numpy()
                    
                        
                    return final_embeddings
            else:
                print(f"Invalid image object type: {type(image_obj)}")
                return None
        except Exception as e:
            print(f"Error creating image embedding: {str(e)}")
            return None

    #i wrote this function to recursively check the dataset and seperated the dataset
    def process_dataset(self,dataset_name,dataset,mul_field=[], is_conversation=False, is_check=False, is_regular=True,Tokenizing=False):
        
        dataset_keys = dataset.features.keys()

        # First level check - initial dataset inspection
        if not is_check and not is_conversation:
            # Use regex to check for conversations column
            print(f"first level check - is conversation:{is_conversation}")
            if any(re.search(self.CONVERSATION_PATTERN, key, re.IGNORECASE) for key in dataset_keys):
                is_conversation = True
            is_check = True
            return self.process_dataset(dataset_name=dataset_name,dataset=dataset,mul_field=mul_field, is_conversation=is_conversation, is_check=is_check,Tokenizing=Tokenizing)

        # Second level check - conversation confirm
        elif is_check and is_conversation:
            # print("Found conversations type dataset with 'conversations' column")
            print(f"second level check - is conversation:{is_conversation}")
            #control what being returned as embed or not
            return self.seperated_data(dataset_name=dataset_name,dataset=dataset,keys='conversations',mul_field=mul_field,Tokenizing=Tokenizing)

        # Third level check - regular dataset processing
        elif is_check and not is_conversation:
            
            if is_regular:
                print("Processing regular dataset")
                # Use regex patterns for message and text columns
                for pattern in self.MESSAGE_PATTERNS:
                    matching_keys = [key for key in dataset_keys if re.search(pattern, key, re.IGNORECASE)]
                    if matching_keys:
                        data_info = dataset[matching_keys[0]]
                        # print(f"Found conversation type dataset with '{matching_keys[0]}' column")
                        if isinstance(data_info, list):
                            return self.seperated_data(dataset_name=dataset_name,dataset=dataset,keys=matching_keys[0],mul_field=mul_field,Tokenizing=Tokenizing)
                        else:
                            print("Trying to format irregular dataset because of no list type")
                            return self.process_dataset(dataset_name=dataset_name,dataset=dataset,mul_field=mul_field, is_conversation=is_conversation, is_check=is_check, is_regular=False,Tokenizing=Tokenizing)
                return self.process_dataset(dataset_name=dataset_name,dataset=dataset,mul_field=mul_field, is_conversation=is_conversation, is_check=is_check, is_regular=False,Tokenizing=Tokenizing)

            # Fourth level check - irregular dataset processing
            if not is_regular:
                print(f"Processing irregular dataset")
                
                dict_list = {"conversations":[]}
                
                
                try:
                    for patterns in self.POTENTIAL_COLUMNS_PATTERNS:
                        # Find matching columns using regex
                        matching_cols_0 = [key for key in dataset_keys if re.search(patterns[0], key, re.IGNORECASE)]
                        matching_cols_1 = [key for key in dataset_keys if re.search(patterns[1], key, re.IGNORECASE)]
                        matching_cols_2 = [key for key in dataset_keys if re.search(patterns[2], key, re.IGNORECASE)]
                        matching_cols_3 = [key for key in dataset_keys if re.search(patterns[3], key, re.IGNORECASE)]
                        matching_cols_4 = [key for key in dataset_keys if re.search(patterns[4], key, re.IGNORECASE)]
                        matching_cols_5 = [key for key in dataset_keys if re.search(patterns[5], key, re.IGNORECASE)]
                        matching_cols_6 = [key for key in dataset_keys if re.search(patterns[6], key, re.IGNORECASE)]
                        matching_cols_7 = [key for key in dataset_keys if re.search(patterns[7], key, re.IGNORECASE)]
                        matching_cols_8 = [key for key in dataset_keys if re.search(patterns[8], key, re.IGNORECASE)]
                        
                        # print(f"matching_cols_0: {matching_cols_0}, matching_cols_1: {matching_cols_1}, matching_cols_2: {matching_cols_2}, matching_cols_3: {matching_cols_3}, matching_cols_4: {matching_cols_4}, matching_cols_5: {matching_cols_5}, matching_cols_6: {matching_cols_6}, matching_cols_7: {matching_cols_7}")

                        if matching_cols_0 and matching_cols_1 and matching_cols_2:
                            #instruction data auto assign role
                            print(f"found {matching_cols_0[0]} and {matching_cols_1[0]} and {matching_cols_2[0]}")
                            for user_q, asist_a, instruction in zip(dataset[matching_cols_0[0]], dataset[matching_cols_1[0]], dataset[matching_cols_2[0]]):
                                message_list = [
                                    {"role": "system", "content": instruction},
                                    {"role": "user", "content": user_q},
                                    {"role": "assistant", "content": asist_a}
                                ]
                                dict_list["conversations"].append(message_list)
                        
                        elif matching_cols_0 and matching_cols_1:
                            #chat data auto assign role
                            print(f"found {matching_cols_0[0]} and {matching_cols_1[0]}")
                            for user_q, asist_a in zip(dataset[matching_cols_0[0]], dataset[matching_cols_1[0]]):
                                message_list = [
                                    {"role": "user", "content": user_q},
                                    {"role": "assistant", "content": asist_a}
                                ]
                                dict_list["conversations"].append(message_list)
                        
                        elif matching_cols_3 and matching_cols_4:
                            #chosen reject instruction
                            print(f"found {matching_cols_3[0]} and {matching_cols_4[0]}")
                            for chosen, rejected in zip(dataset[matching_cols_3[0]], dataset[matching_cols_4[0]]):
                                message_list = [
                                    {"role": "user", "content": chosen},
                                    {"role": "assistant", "content": rejected}
                                ]
                                dict_list["conversations"].append(message_list)
                        
                        elif matching_cols_5 and matching_cols_6:
                            #role text instruction
                            print(f"found {matching_cols_5[0]} and {matching_cols_6[0]}")
                            for role, text in zip(dataset[matching_cols_5[0]], dataset[matching_cols_6[0]]):
                                message_list = [{"role": role, "content": text}]
                                dict_list["conversations"].append(message_list)
                        elif matching_cols_7:
                            #single text instruction
                            print(f"found {matching_cols_7[0]}")
                            for text in dataset[matching_cols_7[0]]:
                                message_list = [{"role": "user", "content": "What in this images?"},
                                                {"role":"assistant","content":text}]
                                dict_list["conversations"].append(message_list)
                        elif matching_cols_8:
                            #single text instruction
                            print(f"found {matching_cols_8[0]}")
                            for text in dataset[matching_cols_8[0]]:
                                message_list = [{"role": "user", "content": "What in this images?"},
                                                {"role":"assistant","content":text}]
                                dict_list["conversations"].append(message_list)
                        else:
                            print(f"No matching columns found for irregular dataset")
                            return
                    # Handle multimodal columns with regex
                    if mul_field and len(mul_field) > 0:
                        print(f"Irregular dataset found multimodal column: {mul_field}")
                        for mul in mul_field:
                            dict_list[mul] = []
                            data_list = []
                            for idx,data in enumerate(dataset[mul]):
                                if data is not None:
                                    data_list.append(data)
                                    # dict_list['conversations'][idx][0]['content'] += f" <images>{data}</images>"
                                    
                            dict_list[mul] = data_list
                    # Create dataset with both conversations and multimodal data
                    print(f"Creating dataset with fields: {list(dict_list.keys())}")
                    dataset_maker = Dataset.from_dict(dict_list)
                    return self.process_dataset(dataset_name=dataset_name,dataset=dataset_maker,is_conversation=False,is_check=False,mul_field=mul_field,Tokenizing=Tokenizing)
                except Exception as e:
                    print(f"Error processing irregular dataset: {str(e)}")
                    return None
        # If we reach here, return None to indicate no valid processing path was found
        print("Warning: No valid processing path found for the dataset")
        return None
    
    
    
    def format_message(self, message):
        formatted_chat = self.template.render(messages=message)
        # print(f"Formatted chat message:{formatted_chat}")
        return formatted_chat
    
    def prepare_dataset(self, dataset_name, dataset, max_length=1000,Tokenizing=False):
        formatted = None

        # Tokenizing
        if Tokenizing:
            print("formatting datasets to embedding datasets")
            try:
                first_example = dataset
                available_fields = list(first_example.features.keys())
                
                def process_examples():
                    mul_field = []
                    for pattern in self.MULTIMODAL_PATTERNS:
                        matching_keys = [key for key in available_fields if re.search(pattern, key, re.IGNORECASE)]
                        if matching_keys:
                            mul_field.extend(matching_keys)

                    
                    if mul_field and len(mul_field) > 0:
                        print("detect multimodal fields",mul_field)
                        formatted = self.process_dataset(
                            dataset_name=dataset_name,
                            dataset=dataset,
                            is_conversation=False,
                            is_check=False,
                            mul_field=mul_field,
                            Tokenizing=Tokenizing
                        )
                    else:
                        formatted = self.process_dataset(
                            dataset_name=dataset_name,
                            dataset=dataset,
                            is_conversation=False,
                            is_check=False,
                            Tokenizing=Tokenizing
                        )
                    
                    if formatted is None:
                        return None
                    
                    return formatted
                
                # Process the dataset if included multimodal data it not supporting batching
                tokenized_dataset = process_examples()
                # Filter out None values
                # tokenized_dataset = tokenized_dataset.filter(lambda x: x is not None)
                
                return tokenized_dataset
                
            except Exception as e:
                print(f"{Fore.RED}Error preparing dataset: {str(e)}{Style.RESET_ALL}")
                raise

        # not tokenize right away
        else:
            formatted = None
            print("formatting datasets to templated datasets")
            try:
                first_example = dataset
                available_fields = list(first_example.features.keys())
                
                mul_field = []
                for pattern in self.MULTIMODAL_PATTERNS:
                    matching_keys = [key for key in available_fields if re.search(pattern, key, re.IGNORECASE)]
                    if matching_keys:
                        mul_field.extend(matching_keys)
                
                if mul_field and len(mul_field) > 0:
                    formatted = self.process_dataset(
                        dataset_name=dataset_name,
                        dataset=first_example,
                        is_conversation=False,
                        is_check=False,
                        mul_field=mul_field,
                        Tokenizing=Tokenizing
                    )
                else:
                    formatted = self.process_dataset(
                        dataset_name=dataset_name,
                        dataset=first_example,
                        is_conversation=False,
                        is_check=False,
                        Tokenizing=Tokenizing
                    )
                
                if formatted is None:
                    return None
                
                return formatted
                
            except Exception as e:
                print(f"{Fore.RED}Error preparing dataset: {str(e)}{Style.RESET_ALL}")
                raise
