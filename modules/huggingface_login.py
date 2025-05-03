import os
from typing import Optional
import logging
from huggingface_hub import login, HfApi,hf_hub_download

import torch
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class HuggingFaceLogin:
    def __init__(self, token: Optional[str] = None):
    
        self.token = token or os.getenv('HUGGING_FACE_TOKEN')
        if not self.token:
            raise ValueError("No token provided. Please provide a token or set HUGGING_FACE_TOKEN environment variable.")
        
        try:
            self.api = HfApi()
            self.hf_download = hf_hub_download
            self._login()
            # Initialize PyTorch with proper settings
            torch.set_num_threads(1)  # Prevent threading issues
        except Exception as e:
            logging.error(f"Error initializing HuggingFaceLogin: {str(e)}")
            raise
    
    def _login(self) -> None:
        try:
            login(token=self.token)
        except Exception as e:
            logging.error(f"Login failed: {str(e)}")
            raise
  