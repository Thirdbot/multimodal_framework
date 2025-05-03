# --base model using transformers pipeline

# --base model architecture change

# --fine-tune and merge with base-model through finetuning_model.py with custom multimodal embedding

#  -- use a model distillation method to distill the base-model into a smaller model

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Use a pipeline as a high-level helper
from transformers import pipeline


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/Pythia-Chat-Base-7B")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/Pythia-Chat-Base-7B")

#pip install flash-attn

class PythiaModel:
    def __init__(self, model_name: str = "togethercomputer/Pythia-Chat-Base-7B"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # Initialize pipeline for easy inference
        self.text_generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **kwargs
    ) -> str:
        """
        Generate text based on the input prompt.
        
        Args:
            prompt (str): Input text prompt
            max_length (int): Maximum length of generated text
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter
            top_k (int): Top-k sampling parameter
            num_return_sequences (int): Number of sequences to generate
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text
        """
        # Format the prompt for Pythia
        formatted_prompt = f"Human: {prompt}\nAssistant:"
        
        # Generate text
        outputs = self.text_generator(
            formatted_prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # Extract and clean the response
        response = outputs[0]['generated_text']
        response = response.split("Assistant:")[-1].strip()
        return response

    def chat(
        self,
        message: str,
        history: list = None,
        max_length: int = 512,
        **kwargs
    ) -> str:
        """
        Generate a chat response based on the message and conversation history.
        
        Args:
            message (str): Current message
            history (list, optional): Conversation history
            max_length (int): Maximum length of generated response
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated response
        """
        if history is None:
            history = []
            
        # Format conversation history
        conversation = ""
        for user_msg, assistant_msg in history:
            conversation += f"Human: {user_msg}\nAssistant: {assistant_msg}\n"
        conversation += f"Human: {message}\nAssistant:"
        
        # Generate response
        response = self.generate(
            conversation,
            max_length=max_length,
            **kwargs
        )
        
        return response

# Initialize the model
pythia_model = PythiaModel()

# Example usage
if __name__ == "__main__":
    # Simple text generation
    print(pythia_model.device)
    prompt = "What is the capital of France?"
    response = pythia_model.generate(prompt)
    print("Generated text:", response)
    
    # Chat example
    message = "Tell me about artificial intelligence"
    chat_response = pythia_model.chat(message)
    print("\nChat response:", chat_response)