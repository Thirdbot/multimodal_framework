import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from login.huggingface_login import HuggingFaceLogin
from transformers import BertTokenizer, BertModel

hf = HuggingFaceLogin()



# #multilingual
# model = BertModel.from_pretrained('bert-base-multilingual-cased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')




# text = "สวัสดีครับ"
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# 





#What To do

#this just addon.
# create fetch for dataset stoff
#create json file store datasets properties
#create class to finetune model with its compatible dataset Multiple time and can be chain function

#main stuff
#create event loop for use input and model input (should recieve multiple input type data as sametime)
#create attention between event loop for filter unwanting data so runtime not interfere
#create embbeding and en-router-attention with de-router-attention (shared attention or embed)
#create function feed input from router to encoder_model or decoder model
#create function to display output by model output from router_attention
#####note output from model should be stream into input of model_input instead of user_input or its model input for inteferencing
#####the data should be on eventloop instead of model loop so crack that 1 bit llms



