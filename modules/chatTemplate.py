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

# Set environment variables for better performance
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class ChatTemplate:
    """Class for handling chat templates and conversation formatting"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        template: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        
        # Initialize tokenizer and model if model_name is provided
        if model_name:
            self.initialize_model(model_name, device)
        
        # Set default template if none provided
        self.default_template = template or """{% for message in messages %}
            {% if message['role'] == 'user' %}
                {{ message['content'] }}
            {% elif message['role'] == 'assistant' %}
                {{ message['content'] }}
            {% endif %}
        {% endfor %}"""
        
        if self.tokenizer:
            self.tokenizer.chat_template = self.default_template
    
    def initialize_model(self, model_name: str, device: str = "auto"):
        """Initialize model and tokenizer"""
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        self.tokenizer.chat_template = self.default_template
    
    def format_conversation(
        self,
        messages: Union[List[Dict[str, str]], Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False
    ) -> Union[str, Dict]:
        """Format a conversation using the chat template"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Call initialize_model() first.")
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt
        )
    
    def load_and_format_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        conversation_field: str = "conversations"
    ):
        """Load and format a dataset using the chat template"""
        dataset = load_dataset(dataset_name, split=split)
        
        # Format the conversations
        dataset = dataset.map(
            lambda x: {
                "formatted_chat": self.format_conversation(x[conversation_field])
            }
        )
        
        return dataset
    
    def set_template(self, template: str):
        """Set a custom chat template"""
        self.default_template = template
        if self.tokenizer:
            self.tokenizer.chat_template = template
    
    def get_template(self) -> str:
        """Get the current chat template"""
        return self.default_template
    
    def is_chat_dataset(self, dataset) -> bool:
        """Check if a dataset contains chat/conversation data"""
        first_example = dataset["train"][0] if "train" in dataset else dataset[0]
        return "conversations" in first_example or "messages" in first_example
    
    def get_conversation_field(self, dataset) -> Optional[str]:
        """Get the field name containing conversations"""
        first_example = dataset["train"][0] if "train" in dataset else dataset[0]
        if "conversations" in first_example:
            return "conversations"
        elif "messages" in first_example:
            return "messages"
        return None

def main():
    # Example usage
    chat_template = ChatTemplate(
        model_name="beatajackowska/DialoGPT-RickBot",
        template="""{% for message in messages %}
            {% if message['role'] == 'user' %}
                User: {{ message['content'] }}
            {% elif message['role'] == 'assistant' %}
                Assistant: {{ message['content'] }}
            {% endif %}
        {% endfor %}"""
    )
    
    # Load and format dataset
    dataset = chat_template.load_and_format_dataset(
        "theneuralmaze/rick-and-morty-transcripts-sharegpt"
    )
    
    # Print first formatted conversation
    print("First formatted conversation:")
    print(dataset['formatted_chat'][0])

if __name__ == "__main__":
    main()