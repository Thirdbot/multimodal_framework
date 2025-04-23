
from datasets import load_dataset

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


dataset = load_dataset("oscar-corpus/OSCAR-2201",
                        language="th", 
                        streaming=True, # optional
                        split="train",
                        trust_remote_code=True) # optional, but the dataset only has a train split

for d in dataset:
    print(d) # prints documents
