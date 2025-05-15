from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
class Chainpipe:
    def __init__(self):
        self.hf_model = None

    def chat_template(self,customtemplate):
        return ChatPromptTemplate(customtemplate)
    
    def load_model_chain(self,model):
        pipe = pipeline(model=model)
        hf = HuggingFacePipeline(pipeline=pipe)
        self.hf_model = hf
        return hf
