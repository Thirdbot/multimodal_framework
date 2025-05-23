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

# Initialize colorama
init(autoreset=True)

class ChatTemplate:
    """Class for handling chat templates and conversation formatting"""
    
    def __init__(self, chainpipe, tokenizer=None, model_name=None, template=None):
        self.chainpipe = chainpipe
        self.tokenizer = tokenizer
        self.prompt = [
            dict({  
                "role": "user",
                "content": [
                    {"text": ""},
                    {"image": ""}
                ]
            }),
            dict({
                "role": "human",
                "content": [
                    {"text": ""},
                    {"image": ""}
                ]
            }),
            dict({
                "role": "assistant",
                "content": [{"text": ""}]
            }),
            dict({
                "role": "system",
                "content": [{"text": ""}]
            })
        ]
        
        self.prompt_map = [key.get('role') for key in self.prompt]
        
        # Create a mapping of role to content types
        self.prompt_map_content = {}
        for prompt in self.prompt:
            role = prompt['role']
            content_types = []
            for content in prompt['content']:
                content_types.append(list(content.keys())[0]) 
            self.prompt_map_content[role] = content_types
            
        print(f"Prompt map content: {self.prompt_map_content}")
    
    def format_conversation(self, dataset_name=None, conversation=None, mul_field=None, ex_data=None, is_train=True):
        """Format a single conversation into a string"""
        try:
            if isinstance(conversation, str):
                return conversation
            
            possible_keys = [('from','value'),('role','content'),('user','text'),('sender','message'),('author','body')]
            str_formatted = ""
    
            
            self.prompt = [
                dict({  
                    "role": "user",
                    "content": [
                        {"text": ""},
                        {"image": ""}
                    ]
                }),
                dict({
                    "role": "human",
                    "content": [
                        {"text": ""},
                        {"image": ""}
                    ]
                }),
                dict({
                    "role": "assistant",
                    "content": [{"text": ""}]
                }),
                dict({
                    "role": "system",
                    "content": [{"text": ""}]
                })
            ]
            
            # Build conversation history
            prompt_text = ""
            for message in conversation:
                if isinstance(message, dict):
                    prompt_text = ""
                    # Handle the new message format
                    if is_train:
                        for key, value in possible_keys:
                            try:
                                # valid_array = message.get(key,None)
                                # if "system" in valid_array:
                                #     #instruction dataset
                                #     pass
                                
                                    #chat converation without system
                                    role = message[key]
                                    content = message[value]
                                    
                                    if role == 'gpt':
                                        role = 'assistant'
                                    
                                    if role not in self.prompt_map:
                                        continue
                                        
                                    role_idx = self.prompt_map.index(role)
                                    content_types = self.prompt_map_content[role]
                                    
                                    # Handle content as a list of dicts
                                    if isinstance(content, list):
                                        for content_item in content:
                                            for field, value in content_item.items():
                                                if field in content_types:
                                                    field_idx = content_types.index(field)
                                                    self.prompt[role_idx]['content'][field_idx][field] = value
                                                    print(f"{Fore.CYAN}Updated {field} for {role}: {value}{Style.RESET_ALL}")
                                    else:
                                        # Handle single content value
                                        field_idx = content_types.index('text')
                                        self.prompt[role_idx]['content'][field_idx]['text'] = content
                                        print(f"{Fore.CYAN}Updated text for {role}: {content}{Style.RESET_ALL}")
                                
                                    # Handle multimodal data if present
                                    if mul_field is not None and ex_data is not None and mul_field in content_types:
                                        field_idx = content_types.index(mul_field)
                                        self.prompt[role_idx]['content'][field_idx][mul_field] = ex_data
                                        
                                    formatted_prompt = self.chainpipe.chat_template(self.prompt)
                                    
                                    for msg in formatted_prompt:
                                        prompt_text += f"Role: {msg.__class__.__name__},Content: {msg.content}\n"
                                #  print(f"{Fore.CYAN}Formatted prompt:{Style.RESET_ALL}\n{self.prompt}")
                                    print(f"{Fore.CYAN}Formatted prompt:{Style.RESET_ALL}\n{prompt_text}")
                                    
                                
                            
                            except Exception as e:
                                continue
                    
                    if not is_train:
                        role = message.get('role', message.get('from', ''))
                        content = message.get('content', message.get('value', ''))
                        
                        if role in ['user', 'human']:
                            conversation_text += f"Human: {content}\n"
                        elif role in ['assistant', 'gpt']:
                            conversation_text += f"Assistant: {content}\n"
                
                        # Add system prompt and conversation history
                        # formatted_prompt = f"{system_prompt}\n\n" if system_prompt else ""
                        formatted_prompt += conversation_text
                        formatted_prompt += "Assistant:"
                        formatted_prompt = self.chainpipe.chat_template(formatted_prompt)
                        # print(f"{Fore.CYAN}Formatted prompt:{Style.RESET_ALL}\n{formatted_prompt}")
                        
            
            return prompt_text
            
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Error formatting conversation: {str(e)}{Style.RESET_ALL}")
            return str(conversation)
    
    def prepare_dataset(self, dataset_name, dataset, max_length=384):
        """
        Prepare a dataset for training by formatting conversations and tokenizing
        Args:
            dataset: HuggingFace dataset
            max_length: Maximum sequence length
        Returns:
            Tokenized dataset ready for training
        """
        try:
            # Check if this is a chat/conversation dataset
            first_example = dataset["train"][0] if "train" in dataset else dataset[0]
            available_fields = list(first_example.keys())
            
        
            for field in ["conversations", "messages", "chat","prompt"]:
                if field in available_fields:
                    conv_field = field
                    break
            
            if conv_field is None:
                print(f"{Fore.YELLOW}No conversation field found, using first available field{Style.RESET_ALL}")
                conv_field = available_fields[0]
            
            def process_examples(examples, batch_size=32):  # Increased batch size for CPU processing
                #since it outer column
                multimodal_fields = ['image','audio','video']
                preference_field = ['chosen','rejected']
        
                mul_field = None
                for field in multimodal_fields:
                    if field in available_fields:
                        mul_field = field
                        break
                    
                # Format conversations in batches
                formatted_texts = []
                if mul_field is not None:
                    # Process multimodal data in batches
                    for i in range(0, len(examples[conv_field]), batch_size):
                        batch_convs = examples[conv_field][i:i + batch_size]
                        batch_data = examples[mul_field][i:i + batch_size]
                        
                        batch_formatted = []
                        for conv, ex_data in zip(batch_convs, batch_data):
                            formatted = self.format_conversation(dataset_name, conv, mul_field, ex_data)
                            batch_formatted.append(formatted)
                        formatted_texts.extend(batch_formatted)
                else:
                    # Process regular conversations in batches
                    for i in range(0, len(examples[conv_field]), batch_size):
                        batch_convs = examples[conv_field][i:i + batch_size]
                        
                        batch_formatted = []
                        for conv in batch_convs:
                            formatted = self.format_conversation(dataset_name, conv)
                            batch_formatted.append(formatted)
                        formatted_texts.extend(batch_formatted)
                
                # Tokenize on CPU with larger batch size
                tokenized = self.tokenizer(
                    formatted_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                
                return tokenized
            
            # Process the dataset with CPU-based batched processing
            tokenized_dataset = dataset.map(
                process_examples,
                batched=True,
                batch_size=32,  # Larger batch size for CPU processing
                remove_columns=dataset["train"].column_names if "train" in dataset else dataset.column_names,
                num_proc=4  # Increased number of processes for CPU
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

def main():
    # Example usage
    try:
        # Initialize with model name
        chat_template = ChatTemplate(
            model_name="gpt2",
            template="""{% for message in messages %}
                {% if message['role'] == 'user' %}
                    User: {{ message['content'] }}
                {% elif message['role'] == 'assistant' %}
                    Assistant: {{ message['content'] }}
                {% endif %}
            {% endfor %}"""
        )
        
        # Example conversation
        conversation = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        # Format and tokenize
        formatted = chat_template.format_conversation(conversation,is_train=False)
        tokenized = chat_template.tokenize_text(formatted)
        
        print("Formatted conversation:")
        print(formatted)
        print("\nTokenized output:")
        print(tokenized)
        
    except Exception as e:
        print(f"{Fore.RED}Error in main: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()