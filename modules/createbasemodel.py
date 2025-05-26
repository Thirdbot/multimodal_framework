import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#for create base model to be use for finetuning and customize architecture
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if __name__ == "__main__":
    # Load the model and tokenizer
    model_name = "beatajackowska/DialoGPT-RickBot"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load the dataset type 1 Conversations column or instruction column
    dataset = load_dataset("theneuralmaze/rick-and-morty-transcripts-sharegpt", split="train")
    
    #load the dataset type 2 regular text columns has an instruction data just like the dataset 1 but not naming its as conversations
    # dataset = load_dataset("Menlo/high-quality-text-only-instruction", "default", split="train")
    
    #load the dataset type 3 regular text columns seperated user and assistant with instruction data
    # dataset = load_dataset("alexgshaw/natural-instructions-prompt-rewards", split="train")
    #load the dataset type 4 regular text columns seperated user and assistant without instruction data
    # dataset = load_dataset("AnonymousSub/MedQuAD_47441_Context_Question_Answer_Triples", split="train")
    
    #instruction seperate column that has chosen and rejected data
    # dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    
    # Print dataset information
    # print("\nDataset structure:")
    # print(dataset)
    
    # # Print a sample from the dataset
    # print("\nSample conversation:")
    # print(dataset['train'][0])
    
    
    
   
    
    
    
    
    
    
   
    # # Print model configuration
    # print("\nModel configuration:")
    # print(f"Model type: {model.config.model_type}")
    # print(f"Vocabulary size: {model.config.vocab_size}")
    # print(f"Hidden size: {model.config.hidden_size}")
    # print(f"Number of layers: {model.config.num_hidden_layers}")