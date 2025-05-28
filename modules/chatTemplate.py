import os
# import torch
from typing import List, Dict, Optional, Union
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from colorama import Fore, Style, init
import zipfile
import torch
# Set environment variables for better performance
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import re
from PIL import Image
from pathlib import Path
# Initialize colorama
init(autoreset=True)

from tqdm import tqdm


from transformers import AutoImageProcessor, AutoModel
from matplotlib.image import imread
# from pydub import AudioSegment
import numpy as np
import pandas as pd

import torch.nn.functional as F
class ChatTemplate:
    """Class for handling chat templates and conversation formatting"""
    
    def __init__(self, chainpipe, tokenizer=None, model_name=None, template=None):
        self.chainpipe = chainpipe
        self.tokenizer = tokenizer
       
        self.img_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.img_model = AutoModel.from_pretrained("google/vit-base-patch16-224")
        
        self.sentence_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-cos-v5")
        self.sentence_model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-cos-v5")
    
    def seperated_data(self,dataset_name,dataset,keys,mul_field=None,return_embedded_dataset=False):
        if not return_embedded_dataset:
            dataset_formated = {"text":[]}
            get_keys = None
            print(f"Processing dataset with key: {keys}")
            
            #chat data
            set_data = dataset.get(f'{keys}')
            if isinstance(set_data, list):
                print(f"Found {len(set_data)} conversations to process")
                for list_data in dataset[f'{keys}']:
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
                    dataset_formated['text'].append(message_list)
                    
                    # Check for multimodal content if needed
                    if mul_field is not None:
                        for msg in message_list:
                            for mul in mul_field:
                                pattern = r"{mul}(.*?)"
                                match = re.findall(pattern,msg['content'])
                                if match:
                                    print(f"Found {mul_field} in content: {match}")
                                    pass
            
            return dataset_formated
        
        elif return_embedded_dataset:
            print(f"Processing embedded dataset")
            total_items = len(dataset[f'{keys}'])
            if mul_field is None:
                print(f"Processing non multimodal conversation")
                # Create new lists to store embedded data
                embedded_messages = []
                embedded_images = []
                
                progress_bar = tqdm(
                        enumerate(dataset[f'{keys}']),
                        total=total_items,
                        desc=f"Embedding non multimodal conversation",
                        unit="item",
                        ncols=100,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                    )
                # Iterate through the dataset with indices
                for index, list_data in progress_bar:
                        try:
                            get_keys = tuple(list_data[0].keys())
                            progress_bar.set_postfix({
                                'status': 'processing',
                                'keys': get_keys
                            })
                            
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
                            
                            try:
                                processed_content = self.get_text_content(role,content,list_data)
                                if processed_content is not None:
                                    embedded_messages.append(processed_content)
                                    progress_bar.set_postfix({
                                        'status': 'success',
                                        'messages': len(embedded_messages)
                                    })
                                else:
                                    progress_bar.set_postfix({
                                        'status': 'warning',
                                        'error': 'None returned'
                                    })
                            except Exception as e:
                                progress_bar.set_postfix({
                                    'status': 'error',
                                    'error': str(e)[:30]
                                })
                                continue
                            
                        except Exception as e:
                            progress_bar.set_postfix({
                                'status': 'error',
                                'error': str(e)[:30]
                            })
                            continue
                
                # Create a new dataset with the embedded data
                print(f"\nCompleted processing {len(embedded_messages)} items")
                new_dataset = Dataset.from_dict({f'{keys}': embedded_messages})
                return new_dataset
            
            elif mul_field is not None:
                for mul in mul_field:
                    # Create new lists to store embedded data
                    embedded_messages = []
                    embedded_images = []
                    
                    # Get total length for progress bar
                    total_items = len(dataset[f'{keys}'])
                    print(f"\nProcessing {total_items} items for {mul} field")
                    
                    # Create progress bar with description
                    progress_bar = tqdm(
                        enumerate(zip(dataset[f'{keys}'], dataset[f'{mul}'])),
                        total=total_items,
                        desc=f"Embedding {mul}",
                        unit="item",
                        ncols=100,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                    )
                    
                    # Iterate through the dataset with indices
                    for index, (list_data, mul_content) in progress_bar:
                        try:
                            get_keys = tuple(list_data[0].keys())
                            progress_bar.set_postfix({
                                'status': 'processing',
                                'keys': get_keys
                            })
                            
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
                            
                            for msg in list_data:
                                msg_get_content = msg[content]
                                if not isinstance(msg_get_content, str):
                                    continue
                                else:
                                    strip_mul = mul.strip('s')
                                    pattern = r"<" + strip_mul + ">(.*?)"
                                    match = re.search(pattern, msg_get_content)
                                    
                                if match:
                                    # last_index_match = match.end()
                                    
                                    user_cut_content = msg_get_content
                                    assistant_cut_content = msg_get_content
                                
                                    if mul == "images":
                                        try:
                                            processed_content, processed_mul_content = self.get_mul_content(
                                                dataset_name, mul_content, user_cut_content, assistant_cut_content, list_data, role, content
                                            )
                                            if processed_content is not None and processed_mul_content is not None:
                                                embedded_messages.append(processed_content)
                                                embedded_images.append(processed_mul_content)
                                                progress_bar.set_postfix({
                                                    'status': 'success',
                                                    'messages': len(embedded_messages),
                                                    'images': len(embedded_images)
                                                })
                                                continue
                                            else:
                                                progress_bar.set_postfix({
                                                    'status': 'warning',
                                                    'error': 'None returned'
                                                })
                                        except Exception as e:
                                            progress_bar.set_postfix({
                                                'status': 'error',
                                                'error': str(e)[:30]
                                            })
                                            continue
                                    elif mul == "audio":
                                        pass
                                    elif mul == "video":
                                        pass
                        except Exception as e:
                            progress_bar.set_postfix({
                                'status': 'error',
                                'error': str(e)[:30]
                            })
                            continue
                    
                    # Create a new dataset with the embedded data
                    print(f"\nCompleted processing {len(embedded_messages)} items")
                    new_dataset = Dataset.from_dict({
                        f'{keys}': embedded_messages,
                        f'{mul}': embedded_images
                    })
                    return new_dataset
    
    def get_text_content(self,role,content,full_data):
        # Process all messages in the conversation
        for index,msg in enumerate(full_data):
            # Replace content with its embedding
            full_data[index][content] = self.text_embedding(msg[content])
            # full_data[index][content] = msg[content]
            
        return full_data
    #make a dataset of mul content
    def get_mul_content(self,dataset_name,mul_content,user_cut_content,assistant_cut_content,full_content,role,content):
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
                processed_messages.append(self.get_text_content(role,content,full_content))
                
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
                processed_messages.append(self.get_text_content(role,content,full_content))
                
                return processed_messages, mul_content_list
        elif isinstance(mul_content,str):
            name_image = mul_content
            mul_embedded = self.get_mul_file(name_image,dataset_name)
            if mul_embedded is not None:
                # Process all messages in the conversation
                processed_messages = []
                processed_messages.append(self.get_text_content(role,content,full_content))
                
                return processed_messages, [mul_embedded]
        else:
            return Exception("Invalid mul_content type")
    
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
            # print(f"Processing file: {file_path}")
            # image
            if file_ext in ['jpg', 'jpeg', 'png']:
                # print(f"Found image: {file_path}")
                try:
                    get_file_obj = zip.open(file_path)
                    # Convert to PIL Image for processing
                    image_obj = Image.open(get_file_obj).convert("RGB")
                    embedded_image = self.image_embedding(image_obj)
                    return embedded_image
                except Exception as e:
                    print(f"Error processing image {file_path}: {str(e)}")
                    return None
            elif file_ext in ['mp3', 'wav']:
                # print(f"Found audio")
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
                    image_obj = Image.open(get_file_obj).convert("RGB")
                    embedded_image = self.image_embedding(image_obj)
                    # print(embedded_image.shape)
                    return embedded_image
                except Exception as e:
                    print(f"Error processing image {file_path}: {str(e)}")
                    return None
            elif file_ext in ['mp3', 'wav']:
                try:
                    # print(f"Found audio")
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
        if isinstance(text, (bytes, bytearray)):
            text = text.decode('utf-8')
        encoded_input = self.sentence_tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.sentence_model(**encoded_input)
            # Perform pooling
            embeddings = self.mean_pooling(outputs, encoded_input['attention_mask'])

            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            return embeddings
    
    def image_embedding(self,image_obj):
        try:
            if isinstance(image_obj, Image.Image):
                # Process image with the model
                inputs = self.img_processor(image_obj, return_tensors="pt")
                outputs = self.img_model(**inputs)
                # Return the pooled output as embedding
                return outputs.pooler_output.detach().numpy()
            else:
                print("Invalid image object type")
                return None
        except Exception as e:
            print(f"Error creating image embedding: {str(e)}")
            return None

    #i wrote this function to recursively check the dataset and seperated the dataset
    def process_dataset(self,dataset_name,dataset,mul_field=None, is_conversation=False, is_check=False, is_regular=True,return_embedded_dataset=False):
        if not return_embedded_dataset:
            # Get dataset keys directly since we're already working with the train split
            dataset_keys = dataset.keys()
            print(f"DEBUG: Dataset keys: {dataset_keys}")
            
            # First level check - initial dataset inspection
            if not is_check and not is_conversation:
                if 'conversations' in dataset_keys:
                    print(f"DEBUG: Found conversations dataset")
                    is_conversation = True
                is_check = True
                return self.process_dataset(dataset_name=dataset_name,dataset=dataset, is_conversation=is_conversation, is_check=is_check)
            
            # Second level check - conversation confirmation
            elif is_check and is_conversation:
                print(f"DEBUG: Processing conversations dataset")
                return self.seperated_data(dataset_name=dataset_name,dataset=dataset,keys='conversations',mul_field=mul_field)
            
            # Third level check - regular dataset processing
            elif is_check and not is_conversation:
                if is_regular:
                    print(f"DEBUG: Processing regular dataset")
                    mis_columns_name = ['messages', 'text']
                    for mis_name in mis_columns_name:
                        if mis_name in dataset_keys:
                            data_info = dataset[mis_name]
                            if isinstance(data_info, list):
                                print(f"DEBUG: Found {mis_name} column with list type")
                                return self.seperated_data(dataset_name=dataset_name,dataset=dataset,keys=mis_name,mul_field=mul_field)
                            else:
                                print(f"DEBUG: Found {mis_name} column with non-list type")
                                print("Trying to format irregular dataset")
                                return self.process_dataset(dataset_name=dataset_name,dataset=dataset, is_conversation=is_conversation, is_check=is_check, is_regular=False)
                    return self.process_dataset(dataset_name=dataset_name,dataset=dataset, is_conversation=is_conversation, is_check=is_check, is_regular=False)
                
                # Fourth level check - irregular dataset processing is seperated instruction column
                if not is_regular:
                    print(f"DEBUG: Processing irregular dataset")
                    potential_columns_name = [(['question','instruction','user','input','Questions'], 
                                            ['answer','response','assistant','output','Answers'],
                                            ['definition','instruction'],
                                            ['chosen'],
                                            ['rejected'],
                                            ['role'],
                                            ['text'])]
                    
                    for potential_columns in potential_columns_name:
                        # Find matching columns for each group
                        matching_cols_0 = [col for col in potential_columns[0] if col in dataset_keys]
                        matching_cols_1 = [col for col in potential_columns[1] if col in dataset_keys]
                        matching_cols_2 = [col for col in potential_columns[2] if col in dataset_keys]
                        matching_cols_3 = [col for col in potential_columns[3] if col in dataset_keys]
                        matching_cols_4 = [col for col in potential_columns[4] if col in dataset_keys]
                        matching_cols_5 = [col for col in potential_columns[5] if col in dataset_keys]
                        matching_cols_6 = [col for col in potential_columns[6] if col in dataset_keys]
                        
                        print(f"DEBUG: Checking potential columns: {potential_columns}")
                        print(f"DEBUG: Found matching columns: {matching_cols_0}, {matching_cols_1}, {matching_cols_2}")

                        if matching_cols_0 and matching_cols_1 and matching_cols_2:
                            dict_list = {"text":[]}
                            #instruction data auto assign role
                            
                            print(f"DEBUG: Processing instruction data")
                            for user_q, asist_a, instruction in zip(dataset[matching_cols_0[0]], dataset[matching_cols_1[0]], dataset[matching_cols_2[0]]):
                                message_list = [
                                    {"role": "system", "content": instruction},
                                    {"role": "user", "content": user_q},
                                    {"role": "assistant", "content": asist_a}
                                ]
                                dict_list["text"].append(message_list)
                            return dict_list
                        
                        elif matching_cols_0 and matching_cols_1:
                            #chat data auto assign role
                        
                            dict_list = {"text": []}
                            for user_q, asist_a in zip(dataset[matching_cols_0[0]], dataset[matching_cols_1[0]]):
                                message_list = [
                                    {"role": "user", "content": user_q},
                                    {"role": "assistant", "content": asist_a}
                                ]
                                dict_list["text"].append(message_list)
                            return dict_list
                        
                        elif matching_cols_3 and matching_cols_4:
                            #chosen reject instruction
                            
                            dict_list = {"text": []}
                            for chosen, rejected in zip(dataset[matching_cols_3[0]], dataset[matching_cols_4[0]]):
                                message_list = [
                                    {"role": "user", "content": chosen},
                                    {"role": "assistant", "content": rejected}
                                ]
                                dict_list["text"].append(message_list)
                            return dict_list
                        
                        elif matching_cols_5 and matching_cols_6:
                            #role text instruction
                            
                            dict_list = {"text": []}
                            for role, text in zip(dataset[matching_cols_5[0]], dataset[matching_cols_6[0]]):
                                message_list = [{"role": role, "content": text}]
                                dict_list["text"].append(message_list)
                            return dict_list
                        else:
                            print(f"Not found any matching columns in {potential_columns}")
                            return {"text": []}
                    print("This dataset cannot be processed")
                
                    return {"text": []}
        elif return_embedded_dataset:
            #change dataset to embedded dataset 3 case like above but return whole embedded dataset structure
            # Get dataset keys directly since we're already working with the train split
            dataset_keys = dataset.features.keys()
            
            # First level check - initial dataset inspection
            if not is_check and not is_conversation:
                if 'conversations' in dataset_keys:
                    # print(f"Conversation Dataset found: {dataset['conversations'][0]} example")
                    is_conversation = True
                is_check = True
                return self.process_dataset(dataset_name=dataset_name,dataset=dataset,mul_field=mul_field, is_conversation=is_conversation, is_check=is_check,return_embedded_dataset=True)
            
            # Second level check - conversation confirmation
            elif is_check and is_conversation:
                return  self.seperated_data(dataset_name=dataset_name,dataset=dataset,keys='conversations',mul_field=mul_field,return_embedded_dataset=True)
            
            # Third level check - regular dataset processing
            elif is_check and not is_conversation:
                if is_regular:
                    # print("Processing regular dataset")
                    mis_columns_name = ['messages', 'text']
                    for mis_name in mis_columns_name:
                        if mis_name in dataset_keys:
                            data_info = dataset[mis_name]
                            if isinstance(data_info, list):
                                return self.seperated_data(dataset_name=dataset_name,dataset=dataset,keys=mis_name,mul_field=mul_field,return_embedded_dataset=True)
                            else:
                                print(f"Found {mis_name} column with non-list type")
                                print("Trying to format irregular dataset")
                                return self.process_dataset(dataset_name=dataset_name,dataset=dataset, is_conversation=is_conversation, is_check=is_check, is_regular=False,return_embedded_dataset=True)
                    return self.process_dataset(dataset_name=dataset_name,dataset=dataset, is_conversation=is_conversation, is_check=is_check, is_regular=False,return_embedded_dataset=True)
                
                # Fourth level check - irregular dataset processing is seperated instruction column
                if not is_regular:
                    # print("Processing irregular dataset")
                    potential_columns_name = [(['question','instruction','user','input','Questions'], 
                                            ['answer','response','assistant','output','Answers'],
                                            ['definition','instruction'],
                                            ['chosen'],
                                            ['rejected'],
                                            ['role'],
                                            ['text'])]
                    
                    for potential_columns in potential_columns_name:
                        # Find matching columns for each group
                        matching_cols_0 = [col for col in potential_columns[0] if col in dataset_keys]
                        matching_cols_1 = [col for col in potential_columns[1] if col in dataset_keys]
                        matching_cols_2 = [col for col in potential_columns[2] if col in dataset_keys]
                        matching_cols_3 = [col for col in potential_columns[3] if col in dataset_keys]
                        matching_cols_4 = [col for col in potential_columns[4] if col in dataset_keys]
                        matching_cols_5 = [col for col in potential_columns[5] if col in dataset_keys]
                        matching_cols_6 = [col for col in potential_columns[6] if col in dataset_keys]
                        
                        

                        if matching_cols_0 and matching_cols_1 and matching_cols_2:
                            dict_list = {"text":[]}
                            #instruction data auto assign role
                            
                            # print(f"Found {matching_cols_0} and {matching_cols_1} and {matching_cols_2} columns")
                            for user_q, asist_a, instruction in zip(dataset[matching_cols_0[0]], dataset[matching_cols_1[0]], dataset[matching_cols_2[0]]):
                                message_list = [
                                    {"role": "system", "content": instruction},
                                    {"role": "user", "content": user_q},
                                    {"role": "assistant", "content": asist_a}
                                ]
                                dict_list["text"].append(message_list)
                            return dict_list
                        
                        elif matching_cols_0 and matching_cols_1:
                            #chat data auto assign role
                        
                            dict_list = {"text": []}
                            for user_q, asist_a in zip(dataset[matching_cols_0[0]], dataset[matching_cols_1[0]]):
                                message_list = [
                                    {"role": "user", "content": user_q},
                                    {"role": "assistant", "content": asist_a}
                                ]
                                dict_list["text"].append(message_list)
                            return dict_list
                        
                        elif matching_cols_3 and matching_cols_4:
                            #chosen reject instruction
                            
                            dict_list = {"text": []}
                            for chosen, rejected in zip(dataset[matching_cols_3[0]], dataset[matching_cols_4[0]]):
                                message_list = [
                                    {"role": "user", "content": chosen},
                                    {"role": "assistant", "content": rejected}
                                ]
                                dict_list["text"].append(message_list)
                            return dict_list
                        
                        elif matching_cols_5 and matching_cols_6:
                            #role text instruction
                            
                            dict_list = {"text": []}
                            for role, text in zip(dataset[matching_cols_5[0]], dataset[matching_cols_6[0]]):
                                message_list = [{"role": role, "content": text}]
                                dict_list["text"].append(message_list)
                            return dict_list
                        else:
                            print(f"Not found any matching columns in {potential_columns}")
                            return {"text": []}
                    print("This dataset cannot be processed")
                
                    return {"text": []}
        
        return False
    
    def format_message(self, message):
        # Format each message with clear role and content separation
        formatted_parts = []
        for msg in message:
            role = msg['role']
            content = msg['content']
            # Add special tokens or markers to clearly separate roles
            if role == 'system' or role == "instruction":
                formatted_parts.append(f"<|system|>\n{content}")
            elif role == 'human' or role == 'user':
                formatted_parts.append(f"<|user|>\n{content}")
            elif role == 'gpt' or role == 'assistant':
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
                    multimodal_fields = ['image','audio','video']
            
                    mul_field = []
                    for field in multimodal_fields:
                        if field in available_fields:
                            mul_field.append(field)
                        
                    if mul_field is not None:
                        print(f"DEBUG: Found multimodal fields: {mul_field}")
                        formatted = self.process_dataset(dataset_name=dataset_name,dataset=examples,is_conversation=False,is_check=False,mul_field=mul_field)
                    else:
                        print(f"DEBUG: No multimodal fields found")
                        formatted = self.process_dataset(dataset_name=dataset_name,dataset=examples,is_conversation=False,is_check=False)
                    
                    # Format all messages in the batch
                    formatted_texts = []
                    for conversation in formatted['text']:
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
                multimodal_fields = ['images','audio','video']
        
                mul_field = []
                for field in multimodal_fields:
                    if field in available_fields:
                        mul_field.append(field)
                    
                if mul_field is not None and len(mul_field) > 0:
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