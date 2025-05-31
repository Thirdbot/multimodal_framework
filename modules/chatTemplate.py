import os
# import torch
from typing import List, Dict, Optional, Union
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from colorama import Fore, Style, init
import zipfile
import torch
import multiprocessing as mp
from functools import partial
# Set environment variables for better performance
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())  # Use all available CPU cores
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import re
from PIL import Image
from pathlib import Path
# Initialize colorama
init(autoreset=True)

from tqdm import tqdm

from joblib import Parallel, delayed


from transformers import AutoImageProcessor, AutoModel
from matplotlib.image import imread
# from pydub import AudioSegment

import torch.nn.functional as F
import numpy as np

from datasets import Dataset

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
         r'^(?:text)$')
    ]
    
    # Role patterns and mappings
    ROLE_PATTERNS = {
        'system': [r'system', r'instruction'],
        'user': [r'user', r'human', r'input'],
        'assistant': [r'assistant', r'gpt', r'output', r'response']
    }
    
    def __init__(self, chainpipe, tokenizer=None, model_name=None, template=None):
        self.chainpipe = chainpipe
        self.tokenizer = tokenizer
       
        # Move model loading to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize image model with ResNet for reliable embeddings
        from torchvision.models import resnet50, ResNet50_Weights
        self.img_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(self.device)
        # Remove the final classification layer
        self.img_model = torch.nn.Sequential(*(list(self.img_model.children())[:-1]))
        self.img_model.eval()
        
        # Initialize image preprocessing
        from torchvision import transforms
        self.img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.sentence_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-cos-v5")
        self.sentence_model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-cos-v5").to(self.device)
    
    def seperated_data(self,dataset_name,dataset,keys,mul_field=None,return_embedded_dataset=False):
        #cut dataset to 1000 temporary
        dataset = dataset[:1000]
        if not return_embedded_dataset:
            #format thinfg back to its original format except for multimodal
            dataset_formated = {"conversations":[]}
            get_keys = None
            print(f"Processing dataset with key: {keys}")
            
            #chat data
            set_data = dataset[keys]
            if isinstance(set_data, list):
                print(f"Found {len(set_data)} conversations to process")
                for list_data in dataset[keys]:
                    get_keys = tuple(list_data[0].keys())
                    message_list = []
                    
                    # Process each message in the conversation
                    for data in list_data:
                        #some conversation have like swap index of role and content(this is just extra check)
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
                            raise ValueError(f"Invalid keys: {get_keys}")
                        
                        role = data[role]
                        content = data[content]
                        format_dict = {"role":role,"content":content}
                        message_list.append(format_dict)
                    
                    # Add the complete conversation to the dataset
                    dataset_formated['conversations'].append(message_list)
                    
                    # Check for multimodal content if needed
                    if mul_field is not None:
                        for msg in message_list:
                            for mul in mul_field:
                                pattern = r"{mul}(.*?)"
                                match = re.findall(pattern,msg['content'])
                                if match:
                                    print(f"Found {mul_field} in content: {match}")
                                    pass
            
            return Dataset.from_dict(dataset_formated)
        
        elif return_embedded_dataset:
            print(f"Processing embedded dataset")
            total_items = len(dataset[keys])
            
            # Reduce batch size for better parallelization
            batch_size = 1000  # Smaller batch size
            batch_list = []
            
            # Process in batches
            for index in range(0, total_items, batch_size):
                end_idx = min(index + batch_size, total_items)
                print(f"Processing batch: {end_idx}/{total_items}", end="\r")
                
                # Create batch dictionary with all required fields
                batch = {
                    keys: dataset[keys][index:end_idx]
                }
                
                # Add multimodal fields if present
                if mul_field is not None:
                    for mul in mul_field:
                        batch[mul] = dataset[mul][index:end_idx]
                
                batch_list.append(batch)
                
            print("\n")
            
            if mul_field is None:
                # Process batches in parallel using all available CPU cores
                n_jobs = min(mp.cpu_count(), 4)  # Limit to 4 processes to avoid memory issues
                print(f"Using {n_jobs} CPU cores for parallel processing")
                
                # Debug multiprocessing
                print(f"Number of batches to process: {len(batch_list)}")
                print(f"First batch size: {len(batch_list[0][keys]) if batch_list else 0}")
                
                # Use multiprocessing with proper backend
                with mp.Pool(processes=n_jobs) as pool:
                    batch_output = list(tqdm(
                        pool.starmap(
                            self.mul_process,
                            [(dataset_name, batch, keys, None) for batch in batch_list]
                        ),
                        total=len(batch_list),
                        desc="Processing batches in parallel"
                    ))
                
                # Flatten the results
                embedded_messages = []
                for batch_result in batch_output:
                    if batch_result is not None:
                        embedded_messages.extend(batch_result)
                
                print(f"\nCompleted processing {len(embedded_messages)} items")
                return Dataset.from_dict({
                    'conversations': embedded_messages
                })
            
            elif mul_field is not None:
                # Process batches in parallel using all available CPU cores
                n_jobs = min(mp.cpu_count(), 4)  # Limit to 4 processes to avoid memory issues
                print(f"Using {n_jobs} CPU cores for parallel processing")
                
                # Debug multiprocessing
                print(f"Number of batches to process: {len(batch_list)}")
                print(f"First batch size: {len(batch_list[0][keys]) if batch_list else 0}")
                
                # Use multiprocessing with proper backend
                with mp.Pool(processes=n_jobs) as pool:
                    results = list(tqdm(
                        pool.starmap(
                            self.mul_process,
                            [(dataset_name, batch, keys, mul_field) for batch in batch_list]
                        ),
                        total=len(batch_list),
                        desc="Processing multimodal batches in parallel"
                    ))
                
                # Process results
                embedded_messages = []
                embedded_images = []
                for batch_messages, batch_images in results:
                    if batch_messages:
                        embedded_messages.extend(batch_messages)
                    if batch_images:
                        embedded_images.extend(batch_images)
                
                print(f"\nCompleted processing {len(embedded_messages)} items")
                
                # Create a list of dictionaries, each containing both text and image data
                combined_data = []
                for msg, img in zip(embedded_messages, embedded_images):
                    combined_data.append({
                        'conversations': msg,
                        'image': img
                    })
                
                return Dataset.from_dict({
                    'conversations': combined_data
                })
                    
                    
    
    
    
    def mul_process(self,dataset_name,dataset,keys,mul_field=None):
        if mul_field is None:
            print(f"Processing non multimodal conversation",end="\r")
            # Create new lists to store embedded data
            batch_data = []
            
            # Get the data for the specified key
            data = dataset[keys]
                
            if not data:
                print(f"Warning: No data found for key {keys}")
                return None
                
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
                
                processed_content = self.get_text_content(role, content, list_data)
                if processed_content is not None:
                    batch_data.append(processed_content)
            
            return batch_data if batch_data else None
        
        elif mul_field is not None:
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
                    print(f"Processing field: {mul}")
                    
                    # Get the data for this field
                    mul_data = dataset[mul]
                    if not mul_data:
                        print(f"Warning: No data found for field {mul}")
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
                                content=content
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
            
            return (embedded_messages, embedded_images) if embedded_messages or embedded_images else (None, None)

    def get_text_content(self, role, content, full_data):
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
                    
                    embedded_content = self.text_embedding(msg[content])
                    # embedded_content = msg[content]
                    if embedded_content is not None:
                        msg[content] = embedded_content

            return full_data
        except Exception as e:
            print(f"Error processing text content: {str(e)}")
            return None
    #make a dataset of mul content
    def get_mul_content(self,dataset_name,mul_content,full_content,role,content):
        #change particular text content inside full content to embedding and every multimodal data to embedding
        if isinstance(mul_content,list):
            if len(mul_content) == 1:
                name_image = mul_content[0]
                mul_content_list = []
                for mul_content_item in mul_content:
                    name_image = mul_content_item
                    mul_embedded = self.get_mul_file(name_image,dataset_name)
                    if mul_embedded is not None:
                        mul_content_list.append(mul_embedded)
                
                # Process all messages in the conversation
                processed_messages = []
                processed_content = self.get_text_content(role, content, full_content)
                if processed_content is not None:
                    processed_messages.append(processed_content)
                
                return processed_messages, mul_content_list
                
            elif len(mul_content) > 1:
                mul_content_list = []
                for mul_content_item in mul_content:
                    name_image = mul_content_item
                    mul_embedded = self.get_mul_file(name_image,dataset_name)
                    if mul_embedded is not None:
                        mul_content_list.append(mul_embedded)
                
                # Process all messages in the conversation
                processed_messages = []
                processed_content = self.get_text_content(role, content, full_content)
                if processed_content is not None:
                    processed_messages.append(processed_content)
                
                return processed_messages, mul_content_list
        elif isinstance(mul_content,str):
            name_image = mul_content
            mul_embedded = self.get_mul_file(name_image,dataset_name)
            if mul_embedded is not None:
                mul_content_list.append(mul_embedded)
                
        else:
            print(f"Invalid mul_content type: {type(mul_content)}")
            return [], []
    
    #get file from local repository
    def get_mul_file(self,data_name,dataset_name):
        short_name = dataset_name.split("/")[-1]
        
        HomePath = Path(__file__).parent.parent.absolute()
        local_dataset_path = HomePath / "repositories" / "datasets" / short_name
        file_in_path = [path for path in Path(local_dataset_path).iterdir() if path.is_file()]
        zip_file_in_path = [path for path in Path(local_dataset_path).iterdir() if path.is_file() and path.suffix == ".zip"]
        folder_in_path = [path for path in Path(local_dataset_path).iterdir() if path.is_dir()]
        if not local_dataset_path.exists():
            print(f"Dataset {dataset_name} not found in local repository")
            return False
        pattern = r"^" + re.escape(data_name) + r"$"
        
        for file in file_in_path:
            if re.match(pattern, file.name):
                # print("file founded")
                return self.read_file(file.name,zip=None)

        for folder in folder_in_path:
            # print("folder founded")
            for file_path in Path(folder).rglob(data_name):
                # print("files founded")
                return self.read_file(file_path,zip=None)

        for zip_file in zip_file_in_path:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for zip_file_name in zip_ref.namelist():
                    if re.match(pattern, zip_file_name):
                        # print("zip file founded")
                        return self.read_file(zip_file_name,zip=zip_ref)
                        
    #get actual data from file
    def read_file(self,file_path,zip=None):
        file_ext = file_path.split('.')[-1].lower()
        if zip:
            # image
            if file_ext in ['jpg', 'jpeg', 'png']:
                try:
                    get_file_obj = zip.open(file_path)
                    # Convert to PIL Image for processing
                    image_obj = Image.open(get_file_obj).convert("RGB")
                    # Resize image to expected size
                    image_obj = image_obj.resize((224, 224), Image.Resampling.LANCZOS)
                    print(f"Processing image from zip: {file_path}")
                    print(f"Image size: {image_obj.size}, mode: {image_obj.mode}")
                    embedded_image = self.image_embedding(image_obj)
                    if embedded_image is not None:
                        print(f"Successfully embedded image from zip: {file_path}")
                    return embedded_image
                except Exception as e:
                    print(f"Error processing image {file_path}: {str(e)}")
                    return None
            elif file_ext in ['mp3', 'wav']:
                try:
                    get_file_obj = zip.open(file_path)
                    # audio_obj = AudioSegment.from_file(get_file_obj)
                    # return self.audio_embedding(audio_obj)
                except Exception as e:
                    print(f"Error processing audio {file_path}: {str(e)}")
                    return None
        else:
            #image
            if file_ext in ['jpg', 'jpeg', 'png']:
                try:
                    get_file_obj = open(file_path, 'rb')
                    # Convert to PIL Image for processing
                    image_obj = Image.open(get_file_obj).convert("RGB")
                    # Resize image to expected size
                    image_obj = image_obj.resize((224, 224), Image.Resampling.LANCZOS)
                    print(f"Processing image: {file_path}")
                    print(f"Image size: {image_obj.size}, mode: {image_obj.mode}")
                    embedded_image = self.image_embedding(image_obj)
                    if embedded_image is not None:
                        print(f"Successfully embedded image: {file_path}")
                    return embedded_image
                except Exception as e:
                    print(f"Error processing image {file_path}: {str(e)}")
                    return None
            elif file_ext in ['mp3', 'wav']:
                try:
                    get_file_obj = open(file_path, 'rb')
                    # audio_obj = AudioSegment.from_file(get_file_obj)
                    # return self.audio_embedding(audio_obj)
                except Exception as e:
                    print(f"Error processing audio {file_path}: {str(e)}")
                    return None
    


    #Mean Pooling - Take average of all tokens
    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def text_embedding(self,text):
        try:
            if isinstance(text, (bytes, bytearray)):
                text = text.decode('utf-8')
            
            # Tokenize the text
            encoded_input = self.sentence_tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors='pt'
            )
            
            # Move inputs to the same device as the model
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Ensure model is in eval mode
            self.sentence_model.eval()
            
            with torch.no_grad():
                # Get model outputs
                outputs = self.sentence_model(**encoded_input)
                
                # Get the last hidden state
                last_hidden_state = outputs.last_hidden_state
                
                # Get attention mask
                attention_mask = encoded_input['attention_mask']
                
                # Mean pooling
                token_embeddings = last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Normalize embeddings
                normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
                
                # Convert to numpy
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
    def process_dataset(self,dataset_name,dataset,mul_field=None, is_conversation=False, is_check=False, is_regular=True,return_embedded_dataset=False):
        dataset_keys = dataset.features.keys()
        
        # First level check - initial dataset inspection
        if not is_check and not is_conversation:
            # Use regex to check for conversations column
            if any(re.search(self.CONVERSATION_PATTERN, key, re.IGNORECASE) for key in dataset_keys):
                is_conversation = True
            is_check = True
            return self.process_dataset(dataset_name=dataset_name,dataset=dataset, is_conversation=is_conversation, is_check=is_check,return_embedded_dataset=return_embedded_dataset)
        
        # Second level check - conversation confirm
        elif is_check and is_conversation:
            print("Found conversations type dataset with 'conversations' column")
            #control what being returned as embed or not
            return self.seperated_data(dataset_name=dataset_name,dataset=dataset,keys='conversations',mul_field=mul_field,return_embedded_dataset=return_embedded_dataset)
        
        # Third level check - regular dataset processing
        elif is_check and not is_conversation:
            if is_regular:
                # Use regex patterns for message and text columns
                for pattern in self.MESSAGE_PATTERNS:
                    matching_keys = [key for key in dataset_keys if re.search(pattern, key, re.IGNORECASE)]
                    if matching_keys:
                        data_info = dataset[matching_keys[0]]
                        print(f"Found conversation type dataset with '{matching_keys[0]}' column")
                        if isinstance(data_info, list):
                            return self.seperated_data(dataset_name=dataset_name,dataset=dataset,keys=matching_keys[0],mul_field=mul_field,return_embedded_dataset=return_embedded_dataset)
                        else:
                            print("Trying to format irregular dataset because of no list type")
                            return self.process_dataset(dataset_name=dataset_name,dataset=dataset, is_conversation=is_conversation, is_check=is_check, is_regular=False,return_embedded_dataset=return_embedded_dataset)
                return self.process_dataset(dataset_name=dataset_name,dataset=dataset, is_conversation=is_conversation, is_check=is_check, is_regular=False,return_embedded_dataset=return_embedded_dataset)
            
            # Fourth level check - irregular dataset processing
            if not is_regular:
                print(f"DEBUG: Processing irregular dataset")
                
                dict_list = {"conversations":[]}
                
                # Handle multimodal columns with regex
                if mul_field and len(mul_field) > 0:
                    print(f"Irregular dataset found multimodal column: {mul_field}")
                    for mul in mul_field:
                        dict_list[mul] = []
                        data_list = []
                        for data in dataset[mul]:
                            if data is not None:
                                data_list.append(data)
                        dict_list[mul] = data_list
                
                for patterns in self.POTENTIAL_COLUMNS_PATTERNS:
                    # Find matching columns using regex
                    matching_cols_0 = [key for key in dataset_keys if re.search(patterns[0], key, re.IGNORECASE)]
                    matching_cols_1 = [key for key in dataset_keys if re.search(patterns[1], key, re.IGNORECASE)]
                    matching_cols_2 = [key for key in dataset_keys if re.search(patterns[2], key, re.IGNORECASE)]
                    matching_cols_3 = [key for key in dataset_keys if re.search(patterns[3], key, re.IGNORECASE)]
                    matching_cols_4 = [key for key in dataset_keys if re.search(patterns[4], key, re.IGNORECASE)]
                    matching_cols_5 = [key for key in dataset_keys if re.search(patterns[5], key, re.IGNORECASE)]
                    matching_cols_6 = [key for key in dataset_keys if re.search(patterns[6], key, re.IGNORECASE)]

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
                        for chosen, rejected in zip(dataset[matching_cols_3[0]], dataset[matching_cols_4[0]]):
                            message_list = [
                                {"role": "user", "content": chosen},
                                {"role": "assistant", "content": rejected}
                            ]
                            dict_list["conversations"].append(message_list)
                    
                    elif matching_cols_5 and matching_cols_6:
                        #role text instruction
                        for role, text in zip(dataset[matching_cols_5[0]], dataset[matching_cols_6[0]]):
                            message_list = [{"role": role, "content": text}]
                            dict_list["conversations"].append(message_list)
                    else:
                        print(f"No matching columns found for irregular dataset")
                        return

                # Create dataset with both conversations and multimodal data
                print(f"Creating dataset with fields: {list(dict_list.keys())}")
                dataset_maker = Dataset.from_dict(dict_list)
                return self.process_dataset(dataset_name=dataset_name,dataset=dataset_maker,is_conversation=False,is_check=False,mul_field=mul_field,return_embedded_dataset=return_embedded_dataset)
        
        # If we reach here, return None to indicate no valid processing path was found
        print("Warning: No valid processing path found for the dataset")
        return None
    
    def format_message(self, message):
        # Format each message with clear role and content separation
        formatted_parts = []
        for msg in message:
            role = msg['role']
            content = msg['content']
            
            # Determine role type using regex patterns
            role_type = None
            for role_type, patterns in self.ROLE_PATTERNS.items():
                if any(re.search(pattern, role, re.IGNORECASE) for pattern in patterns):
                    role_type = role_type
                    break
            
            # Add special tokens or markers to clearly separate roles
            if role_type == 'system':
                formatted_parts.append(f"<|system|>\n{content}")
            elif role_type == 'user':
                formatted_parts.append(f"<|user|>\n{content}")
            elif role_type == 'assistant':
                formatted_parts.append(f"<|assistant|>\n{content}")
            else:
                formatted_parts.append(f"<|{role}|>\n{content}")
        
        # Join all parts with double newlines for clear separation
        return "\n\n".join(formatted_parts)
    
    def prepare_dataset(self, dataset_name, dataset, max_length=1000,return_embedded_dataset=False):
        if not return_embedded_dataset:
            try:
                print(f"DEBUG: Starting prepare_dataset with dataset_name: {dataset_name}")
                print(f"DEBUG: Dataset type: {type(dataset)}")
                first_example = dataset
                available_fields = list(first_example.features.keys())
                print(f"DEBUG: Available fields: {available_fields}")
                
                def process_examples(examples, batch_size=32):  # Increased batch size for CPU processing
                    print(f"DEBUG: Processing examples batch")
                    #since it outer column
                    mul_field = []
                    for pattern in self.MULTIMODAL_PATTERNS:
                        matching_keys = [key for key in available_fields if re.search(pattern, key, re.IGNORECASE)]
                        if matching_keys:
                            mul_field.extend(matching_keys)
                            print(f"DEBUG: Found multimodal field: {matching_keys[0]}")
                        
                    if mul_field and len(mul_field) > 0:
                        print(f"DEBUG: Found multimodal fields: {mul_field}")
                        # Convert LazyBatch to Dataset before processing
                        examples_dataset = Dataset.from_dict({k: examples[k] for k in examples.keys()})
                        formatted = self.process_dataset(dataset_name=dataset_name,dataset=examples_dataset,is_conversation=False,is_check=False,mul_field=mul_field)
                    else:
                        print(f"DEBUG: No multimodal fields found")
                        # Convert LazyBatch to Dataset before processing
                        examples_dataset = Dataset.from_dict({k: examples[k] for k in examples.keys()})
                        formatted = self.process_dataset(dataset_name=dataset_name,dataset=examples_dataset,is_conversation=False,is_check=False)
                    
                    # Format all messages in the batch
                    formatted_texts = []
                    for conversation in formatted['conversations']:
                        formatted_text = self.format_message(conversation)
                        formatted_texts.append(formatted_text)
                    
                    # Tokenize the entire batch at once
                    tokenized = self.tokenizer(
                        formatted_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    )
                    
                    # Convert to lists for dataset compatibility
                    return tokenized
                
                # Process the dataset with batched processing
                print(f"DEBUG: Starting dataset mapping")
                tokenized_dataset = dataset.map(
                    process_examples,
                    batched=True,
                    batch_size=32,
                    remove_columns=dataset.column_names  # Remove original columns
                )
                
                print(f"{Fore.GREEN}Successfully prepared dataset with {len(tokenized_dataset)} examples{Style.RESET_ALL}")
                return tokenized_dataset
                
            except Exception as e:
                print(f"{Fore.RED}Error preparing dataset: {str(e)}{Style.RESET_ALL}")
                raise
        elif return_embedded_dataset:
            try:
                print(f"DEBUG: Starting prepare_dataset with embedded dataset")
                first_example = dataset
                available_fields = list(first_example.features.keys())
                print(f"DEBUG: Available fields for embedded dataset: {available_fields}")
                
                #since it outer column
                mul_field = []
                for pattern in self.MULTIMODAL_PATTERNS:
                    matching_keys = [key for key in available_fields if re.search(pattern, key, re.IGNORECASE)]
                    if matching_keys:
                        mul_field.extend(matching_keys)
                        print(f"DEBUG: Found multimodal field: {matching_keys[0]}")
                    
                if mul_field and len(mul_field) > 0:
                    print(f"DEBUG: Found multimodal fields for embedded dataset: {mul_field}")
                    formatted = self.process_dataset(dataset_name=dataset_name,dataset=first_example,is_conversation=False,is_check=False,mul_field=mul_field,return_embedded_dataset=True)
                else:
                    print(f"DEBUG: No multimodal fields found for embedded dataset")
                    formatted = self.process_dataset(dataset_name=dataset_name,dataset=first_example,is_conversation=False,is_check=False,return_embedded_dataset=True)
                
                return formatted
                
            except Exception as e:
                print(f"{Fore.RED}Error preparing dataset: {str(e)}{Style.RESET_ALL}")
                raise
    
    def get_tokenizer(self):
        """Get the tokenizer instance"""
        return self.tokenizer
    
    def get_chat_template(self):
        """Get the current chat template"""
        return self.tokenizer.chat_template
    
    def set_chat_template(self, template):
        """Set a custom chat template"""
        self.tokenizer.chat_template = template
        print(f"{Fore.CYAN}Set custom chat template{Style.RESET_ALL}")
    
    def tokenize_text(self, text, max_length=384):
        """Tokenize a single text input"""
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def decode_tokens(self, tokens):
        """Decode tokenized input back to text"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)