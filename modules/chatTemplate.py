# class chatTemplate:
#     def __init__(self,messages:list[str],role:list[str]=['user','assistant']):
#         self.messages = messages
#         self.role = role

#     def createTemplate(self):
#         for idx, message in enumerate(self.messages):
#             if message['role'] == 'user':
#                 print(' ')
#             print(message['content'])
#             if not idx == len(self.messages) - 1:  # Check for the last message in the conversation
#                 print('  ')
#         print(eos_token)

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
    
    def __init__(self, tokenizer=None, model_name=None, template=None):
        """
        Initialize ChatTemplate with either a tokenizer or model name
        Args:
            tokenizer: Pre-initialized tokenizer
            model_name: Model name to load tokenizer from
            template: Optional custom chat template
        """
        try:
            if tokenizer is not None:
                self.tokenizer = tokenizer
            elif model_name is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    padding_side="right",
                    truncation_side="right"
                )
            else:
                # Initialize with a default tokenizer if none provided
                #may be use from save model or gpt
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "gpt2",  # Default model
                    trust_remote_code=True,
                    padding_side="right",
                    truncation_side="right"
                )
                print(f"{Fore.YELLOW}Warning: No tokenizer or model_name provided, using default GPT-2 tokenizer{Style.RESET_ALL}")
            
            # Set chat template
            if template is not None:
                self.tokenizer.chat_template = template
                print(f"{Fore.CYAN}Set custom chat template{Style.RESET_ALL}")
            elif not hasattr(self.tokenizer, "chat_template") or self.tokenizer.chat_template is None:
                self.tokenizer.chat_template = self._get_default_chat_template()
                print(f"{Fore.CYAN}Set default chat template{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error initializing ChatTemplate: {str(e)}{Style.RESET_ALL}")
            raise
    
    def _get_default_chat_template(self):
        """Get a default chat template if none is set"""
        return """{% for message in messages %}
            {% if message['role'] == 'user' %}
                User: {{ message['content'] }}
            {% elif message['role'] == 'assistant' %}
                Assistant: {{ message['content'] }}
            {% elif message['role'] == 'system' %}
                System: {{ message['content'] }}
            {% endif %}
        {% endfor %}"""
    
    def format_conversation(self, conversation):
        """Format a single conversation into a string"""
        try:
            if isinstance(conversation, str):
                print(f"{conversation}")
                return conversation
            
            formatted = []
            possible_keys = [('role','content'),('user','text'),('sender','message'),('author','body'),('from','value')]
            for message in conversation:
                if isinstance(message, dict):
                    for keysend,keyrecv in possible_keys:
                        try:
                            role = message[keysend]
                            content = message[keyrecv]
                                   
                        except Exception as e:
                            continue
                        print(f"{role}: {content}")
                        formatted.append(f"{role}: {content}")
                elif isinstance(message, str):
                    print(f"{message}")
                    formatted.append(message)
            
            return "\n".join(formatted)
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Error formatting conversation: {str(e)}{Style.RESET_ALL}")
            return str(conversation)
    
    def prepare_dataset(self, dataset, max_length=384):
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
            
            # Determine conversation field
            conv_field = None
            for field in ["conversations", "messages", "chat"]:
                if field in available_fields:
                    conv_field = field
                    break
            
            if conv_field is None:
                print(f"{Fore.YELLOW}No conversation field found, using first available field{Style.RESET_ALL}")
                conv_field = available_fields[0]
            
            def process_examples(examples):
                # Format conversations
                formatted_texts = []
                for conv in examples[conv_field]:
                    formatted = self.format_conversation(conv)
                    print(f"{formatted}/n")
                    formatted_texts.append(formatted)
                # Tokenize
                tokenized = self.tokenizer(
                    formatted_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                
                return tokenized
            
            # Process the dataset
            tokenized_dataset = dataset.map(
                process_examples,
                batched=True,
                remove_columns=dataset["train"].column_names if "train" in dataset else dataset.column_names,
                num_proc=2
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
        formatted = chat_template.format_conversation(conversation)
        tokenized = chat_template.tokenize_text(formatted)
        
        print("Formatted conversation:")
        print(formatted)
        print("\nTokenized output:")
        print(tokenized)
        
    except Exception as e:
        print(f"{Fore.RED}Error in main: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()