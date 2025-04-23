import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from login.huggingface_login import HuggingFaceLogin
from transformers import BertTokenizer, BertModel

hf = HuggingFaceLogin()


# #multilingual
model = BertModel.from_pretrained('bert-base-multilingual-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')




text = "สวัสดีครับ"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
