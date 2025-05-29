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

# def process_batch(batch_data, dataset_name, keys, mul_field=None):
#     """Helper function for multiprocessing"""
#     if mul_field is None:
#         # Process non-multimodal data
#         batch_data = []
#         for list_data in batch_data:
#             get_keys = tuple(list_data[0].keys())
#             if get_keys[0] in ('role','from'):
#                 role = get_keys[0]
#                 content = get_keys[1]
#             elif get_keys[1] in ('content','value'):
#                 role = get_keys[1]
#                 content = get_keys[0]
#             elif get_keys[0] in ('content','value'):
#                 role = get_keys[1]
#                 content = get_keys[0]
#             elif get_keys[1] in ('role','from'):
#                 role = get_keys[1]
#                 content = get_keys[0]
#             else:
#                 raise ValueError(f"Invalid keys: {get_keys}")
            
#             processed_content = get_text_content(role, content, list_data)
#             if processed_content is not None:
#                 batch_data.append(processed_content)
#         return batch_data
#     else:
#         # Process multimodal data
#         embedded_messages = []
#         embedded_images = []
        
#         # Process each multimodal field
#         for mul in mul_field:
#             try:
#                 # Get the data for this field
#                 mul_data = batch_data[mul]
#                 text_data = batch_data[keys]
                
#                 # Process each item in the dataset
#                 for text_item, mul_item in zip(text_data, mul_data):
#                     try:
#                         # Get the keys for text processing
#                         if isinstance(text_item, dict):
#                             get_keys = tuple(text_item.keys())
#                         elif isinstance(text_item, list) and len(text_item) > 0:
#                             get_keys = tuple(text_item[0].keys())
#                         else:
#                             print(f"Unexpected text item format: {text_item}")
#                             continue
                            
#                         if get_keys[0] in ('role','from'):
#                             role = get_keys[0]
#                             content = get_keys[1]
#                         elif get_keys[1] in ('content','value'):
#                             role = get_keys[1]
#                             content = get_keys[0]
#                         elif get_keys[0] in ('content','value'):
#                             role = get_keys[1]
#                             content = get_keys[0]
#                         elif get_keys[1] in ('role','from'):
#                             role = get_keys[1]
#                             content = get_keys[0]
#                         else:
#                             raise ValueError(f"Invalid keys: {get_keys}")
                        
#                         # Process the multimodal content
#                         processed_messages, processed_images = get_mul_content(
#                             dataset_name=dataset_name,
#                             mul_content=mul_item,
#                             full_content=text_item,
#                             role=role,
#                             content=content
#                         )
                        
#                         if processed_messages:
#                             embedded_messages.extend(processed_messages)
#                         if processed_images:
#                             embedded_images.extend(processed_images)
                            
#                     except Exception as e:
#                         print(f"Error processing item: {str(e)}")
#                         continue
                        
#             except Exception as e:
#                 print(f"Error processing {mul} field: {str(e)}")
#                 continue
        
#         return embedded_messages, embedded_images

class ChatTemplate:
    """Class for handling chat templates and conversation formatting"""
    
    def __init__(self, chainpipe, tokenizer=None, model_name=None, template=None):
        self.chainpipe = chainpipe
        self.tokenizer = tokenizer
       
        # Move model loading to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize image model with CLIP for better image embeddings
        self.img_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.img_model = AutoModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            output_hidden_states=True,
            return_dict=True
        ).to(self.device)
        
        # Ensure model is in eval mode
        self.img_model.eval()
        
        self.sentence_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-cos-v5")
        self.sentence_model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-cos-v5").to(self.device)
    
    def seperated_data(self,dataset_name,dataset,keys,mul_field=None,return_embedded_dataset=False):
        #cut dataset to 1000 temporary
        dataset = dataset[:1000]
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
            total_items = len(dataset[keys])
            
            # Reduce batch size for better parallelization
            batch_size = 100000  # Smaller batch size
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
                new_dataset = Dataset.from_dict({
                    f'{keys}': embedded_messages,
                })
                return new_dataset
            
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
                        'text': msg,
                        'image': img
                    })
                
                new_dataset = Dataset.from_dict({
                    f'{keys}': combined_data
                })
                return new_dataset
                    
                    
    
    
    
    def mul_process(self,dataset_name,dataset,keys,mul_field=None):
        if mul_field is None:
            print(f"Processing non multimodal conversation",end="\r")
            # Create new lists to store embedded data
            batch_data = []
            
            # Get the data for the specified key
            data = dataset.get(keys, [])
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
            text_data = dataset.get(keys, [])
            if not text_data:
                print(f"Warning: No text data found for key {keys}")
                return None, None
            
            # Process each multimodal field
            for mul in mul_field:
                try:
                    print(f"Processing field: {mul}")
                    
                    # Get the data for this field
                    mul_data = dataset.get(mul, [])
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
                    embedded_content = self.text_embedding(msg[content])
                    # print(f"msg: {msg[content]}")
                    if embedded_content is not None:
                        msg[content] = embedded_content
                        # batch_data.append(msg)

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
        if isinstance(text, (bytes, bytearray)):
            text = text.decode('utf-8')
        encoded_input = self.sentence_tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        # Move inputs to the same device as the model
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            outputs = self.sentence_model(**encoded_input)
            # Perform pooling
            embeddings = self.mean_pooling(outputs, encoded_input['attention_mask'])

            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy()  # Move to CPU before returning
    
    def image_embedding(self,image_obj):
        try:
            if isinstance(image_obj, Image.Image):
                # Process image with the model
                inputs = self.img_processor(image_obj, return_tensors="pt")
                # Move inputs to the same device as the model
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Print input stats
                print(f"Input pixel values stats - min: {inputs['pixel_values'].min().item()}, max: {inputs['pixel_values'].max().item()}, mean: {inputs['pixel_values'].mean().item()}")
                
                with torch.no_grad():
                    # Get model outputs
                    outputs = self.img_model(**inputs)
                    
                    # Print all available outputs
                    print("Available outputs:", outputs.keys())
                    
                    # Get the image embeddings
                    image_embeddings = outputs.image_embeds
                    print(f"Image embeddings shape: {image_embeddings.shape}")
                    
                    # Print raw embeddings stats
                    print(f"Raw embeddings stats - min: {image_embeddings.min().item()}, max: {image_embeddings.max().item()}, mean: {image_embeddings.mean().item()}")
                    
                    # Normalize the embeddings
                    normalized_embeddings = F.normalize(image_embeddings, p=2, dim=1)
                    
                    # Print normalized embeddings stats
                    print(f"Normalized embeddings stats - min: {normalized_embeddings.min().item()}, max: {normalized_embeddings.max().item()}, mean: {normalized_embeddings.mean().item()}")
                    
                    # Convert to numpy
                    final_embeddings = normalized_embeddings.detach().cpu().numpy()
                    
                    # Print final embeddings stats
                    print(f"Final embeddings stats - min: {final_embeddings.min()}, max: {final_embeddings.max()}, mean: {final_embeddings.mean()}")
                    
                    # Verify embeddings are not all zeros
                    if np.all(np.abs(final_embeddings) < 1e-6):
                        print("Warning: All zero embeddings detected")
                        return None
                        
                    return final_embeddings
            else:
                print(f"Invalid image object type: {type(image_obj)}")
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
                    multimodal_fields = ['image', 'images', 'audio', 'video']
            
                    mul_field = []
                    for field in multimodal_fields:
                        if field in available_fields:
                            mul_field.append(field)
                            print(f"DEBUG: Found multimodal field: {field}")
                        
                    if mul_field and len(mul_field) > 0:
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
                multimodal_fields = ['image', 'images', 'audio', 'video']
        
                mul_field = []
                for field in multimodal_fields:
                    if field in available_fields:
                        mul_field.append(field)
                        print(f"DEBUG: Found multimodal field: {field}")
                    
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