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
    dataset = load_dataset("theneuralmaze/rick-and-morty-transcripts-sharegpt")
    
    #load the dataset type 2 regular text columns has an instruction data just like the dataset 1 but not naming its as conversations
    # dataset = load_dataset("Menlo/high-quality-text-only-instruction", "default")
    
    #load the dataset type 3 regular text columns seperated user and assistant with instruction data
    # dataset = load_dataset("alexgshaw/natural-instructions-prompt-rewards")
    #load the dataset type 4 regular text columns seperated user and assistant without instruction data
    # dataset = load_dataset("AnonymousSub/MedQuAD_47441_Context_Question_Answer_Triples")
    
    #instruction seperate column that has chosen and rejected data
    # dataset = load_dataset("trl-lib/ultrafeedback_binarized")
    
    # Print dataset information
    # print("\nDataset structure:")
    # print(dataset)
    
    # # Print a sample from the dataset
    # print("\nSample conversation:")
    # print(dataset['train'][0])
    
    
    
    def seperated_data(dataset,keys):
        common_instructor_role = ['system']
        get_keys = None
        print("Processed a conversation")
        
        #chat data
        set_data = dataset['train'].features.get(f'{keys}')
        if isinstance(set_data, list):
            get_keys =tuple( set_data[0].keys())
            print(get_keys)
        for list_data in dataset['train'][f'{keys}']:
            role_list = []
            for data in list_data:
                role = data[get_keys[0]]
                content = data[get_keys[1]]
                
                role_list.append(role)
            
            for role in role_list:
                if role in common_instructor_role:
                    print(f"Found common instructor role: {role}")
                    #instruct
                    break
                else:
                    #chat data
                    print("regular chat data")
                    print(f"Role: {role},Content: {content}")
                    break
            break
        
    def process_dataset(dataset, is_conversation=False, is_check=False, is_regular=True):
        dataset_keys = dataset['train'].features.keys()
        
        # First level check - initial dataset inspection
        if not is_check and not is_conversation:
            if 'conversations' in dataset_keys:
                print(f"Conversation Dataset found: {dataset['train']['conversations'][0]} example")
                is_conversation = True
            is_check = True
            return process_dataset(dataset=dataset, is_conversation=is_conversation, is_check=is_check)
        
        # Second level check - conversation confirmation
        elif is_check and is_conversation:
            seperated_data(dataset=dataset,keys='conversations')
            return True
        
        # Third level check - regular dataset processing
        elif is_check and not is_conversation:
            if is_regular:
                print("Processing regular dataset")
                mis_columns_name = ['messages', 'text']
                for mis_name in mis_columns_name:
                    if mis_name in dataset_keys:
                        data_info = dataset['train'].features.get(mis_name)
                        if isinstance(data_info, list):
                            print(f"Found {mis_name} column with list type")
                            print(data_info[0])
                            
                            #seperated data
                            seperated_data(dataset=dataset,keys=mis_name)
                            
                            return True
                        else:
                            print(f"Found {mis_name} column with non-list type")
                            print("Trying to format irregular dataset")
                            return process_dataset(dataset=dataset, is_conversation=is_conversation, is_check=is_check, is_regular=False)
                return process_dataset(dataset=dataset, is_conversation=is_conversation, is_check=is_check, is_regular=False)
            
            # Fourth level check - irregular dataset processing is seperated instruction column
            if not is_regular:
                print("Processing irregular dataset")
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
                        #instruction data auto assign role
                        print("found instruction data")
                        print(f"Found {matching_cols_0} and {matching_cols_1} and {matching_cols_2} columns")
                        return True, (matching_cols_0, matching_cols_1, matching_cols_2)
                    
                    elif matching_cols_0 and matching_cols_1:
                        #chat data auto assign role
                        print("found chat data")
                        print(f"Found {matching_cols_0} and {matching_cols_1} columns")
                        return True, (matching_cols_0, matching_cols_1)
                    
                    elif matching_cols_3 and matching_cols_4:
                        #chosen reject instruction
                        print("found chosen reject instruction")
                        print(f"Found {matching_cols_3} and {matching_cols_4} columns")
                        return True, (matching_cols_3, matching_cols_4)
                    
                    elif matching_cols_5 and matching_cols_6:
                        #role text instruction
                        print("found role text column chat")
                        print(f"Found {matching_cols_5} and {matching_cols_6} columns")
                        return True, (matching_cols_5, matching_cols_6)
                    else:
                        print(f"Not found any matching columns in {potential_columns}")
                        return False, None
                print("This dataset cannot be processed")
                
               

                return False
        
        return False
    
    process_dataset(dataset=dataset,is_conversation=False,is_check=False)
    
    
    
   
    # # Print model configuration
    # print("\nModel configuration:")
    # print(f"Model type: {model.config.model_type}")
    # print(f"Vocabulary size: {model.config.vocab_size}")
    # print(f"Hidden size: {model.config.hidden_size}")
    # print(f"Number of layers: {model.config.num_hidden_layers}")