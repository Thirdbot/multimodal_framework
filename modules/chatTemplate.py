import os
# import torch
from typing import List, Dict, Optional, Union
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from colorama import Fore, Style, init

# Set environment variables for better performance
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import re

# Initialize colorama
init(autoreset=True)

class ChatTemplate:
    """Class for handling chat templates and conversation formatting"""
    
    def __init__(self, chainpipe, tokenizer=None, model_name=None, template=None):
        self.chainpipe = chainpipe
        self.tokenizer = tokenizer
    
    def seperated_data(self,dataset,keys,mul_field=None):
        dataset_formated = {"text":[]}
        get_keys = None
        # print(f"Processing dataset with key: {keys}")
        
        #chat data
        set_data = dataset.get(f'{keys}')
        if isinstance(set_data, list):
            # print(f"Found {len(set_data)} conversations to process")
            for list_data in dataset[f'{keys}']:
                get_keys = tuple(list_data[0].keys())
                message_list = []
                
                # Process each message in the conversation
                for data in list_data:
                    role = data[get_keys[0]]
                    content = data[get_keys[1]]
                    format_dict = {"role":role,"content":content}
                    message_list.append(format_dict)
                
                # Add the complete conversation to the dataset
                dataset_formated['text'].append(message_list)
                
                # Check for multimodal content if needed
                if mul_field is not None:
                    for msg in message_list:
                        pattern = r"<{mul_field}>(.*?)"
                        match = re.findall(pattern,msg['content'])
                        if match:
                            # print(f"Found {mul_field} in content: {match}")
                            # concatenate real data over tags
                            pass
                            break
        
        return dataset_formated
        
    #i wrote this function to recursively check the dataset and seperated the dataset
    def process_dataset(self,dataset,mul_field=None, is_conversation=False, is_check=False, is_regular=True):
        # Get dataset keys directly since we're already working with the train split
        dataset_keys = dataset.keys()
        
        # First level check - initial dataset inspection
        if not is_check and not is_conversation:
            if 'conversations' in dataset_keys:
                # print(f"Conversation Dataset found: {dataset['conversations'][0]} example")
                is_conversation = True
            is_check = True
            return self.process_dataset(dataset=dataset, is_conversation=is_conversation, is_check=is_check)
        
        # Second level check - conversation confirmation
        elif is_check and is_conversation:
            return  self.seperated_data(dataset=dataset,keys='conversations',mul_field=mul_field)
        
        # Third level check - regular dataset processing
        elif is_check and not is_conversation:
            if is_regular:
                # print("Processing regular dataset")
                mis_columns_name = ['messages', 'text']
                for mis_name in mis_columns_name:
                    if mis_name in dataset_keys:
                        data_info = dataset[mis_name]
                        if isinstance(data_info, list):
                            
                            return self.seperated_data(dataset=dataset,keys=mis_name,mul_field=mul_field)
                        else:
                            print(f"Found {mis_name} column with non-list type")
                            print("Trying to format irregular dataset")
                            return self.process_dataset(dataset=dataset, is_conversation=is_conversation, is_check=is_check, is_regular=False)
                return self.process_dataset(dataset=dataset, is_conversation=is_conversation, is_check=is_check, is_regular=False)
            
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
    
    def prepare_dataset(self, dataset_name, dataset, max_length=384):
        try:
            first_example = dataset
            available_fields = list(first_example.features.keys())
            
            def process_examples(examples, batch_size=32):  # Increased batch size for CPU processing
                #since it outer column
                multimodal_fields = ['image','audio','video']
        
                mul_field = None
                for field in multimodal_fields:
                    if field in available_fields:
                        mul_field = field
                        break
                    
                if mul_field is not None:
                    formatted = self.process_dataset(dataset=examples,is_conversation=False,is_check=False,mul_field=mul_field)
                else:
                    formatted = self.process_dataset(dataset=examples,is_conversation=False,is_check=False)
                
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
                return {
                    "input_ids": tokenized["input_ids"].tolist(),
                    "attention_mask": tokenized["attention_mask"].tolist()
                }
            
            # Process the dataset with batched processing
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