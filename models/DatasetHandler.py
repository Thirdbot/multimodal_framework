
from datasets import load_dataset

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class DatasetHandler:
    def __init__(self, name,language,split):
        self.name = name
        self.language = language
        self.split = split
        self.dataset = load_dataset(name,
                        language=self.language, 
                        streaming=True, # optional
                        split=self.split,
                        trust_remote_code=True) # optional, but the dataset only has a train split


    def get_dataset(self):
        return self.dataset



