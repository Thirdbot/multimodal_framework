import os
from typing import Optional, Union, Dict, Any
import logging
from huggingface_hub import login, HfApi
from datasets import load_dataset
import torch

class HuggingFaceLogin:
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the HuggingFaceLogin class.
        
        Args:
            token (str, optional): Hugging Face API token. If not provided, will look for HUGGING_FACE_TOKEN environment variable.
        """
        self.token = token or os.getenv('HUGGING_FACE_TOKEN')
        if not self.token:
            raise ValueError("No token provided. Please provide a token or set HUGGING_FACE_TOKEN environment variable.")
        
        try:
            self.api = HfApi()
            self._login()
            # Initialize PyTorch with proper settings
            torch.set_num_threads(1)  # Prevent threading issues
        except Exception as e:
            logging.error(f"Error initializing HuggingFaceLogin: {str(e)}")
            raise
    
    def _login(self) -> None:
        """Authenticate with Hugging Face Hub."""
        try:
            login(token=self.token)
        except Exception as e:
            logging.error(f"Login failed: {str(e)}")
            raise
    
    def load_dataset(
        self,
        dataset_name: str,
        split: Optional[Union[str, list]] = None,
        cache_dir: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """
        Load a dataset from Hugging Face Hub.
        
        Args:
            dataset_name (str): Name of the dataset to load
            split (str or list, optional): Split(s) to load
            cache_dir (str, optional): Directory to cache the dataset
            **kwargs: Additional arguments to pass to load_dataset
            
        Returns:
            Dataset object
        """
        try:
            return load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir,
                token=self.token,
                **kwargs
            )
        except Exception as e:
            logging.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            Dict containing dataset information
        """
        try:
            return self.api.dataset_info(dataset_name)
        except Exception as e:
            logging.error(f"Error getting dataset info for {dataset_name}: {str(e)}")
            raise
    
    def list_dataset_splits(self, dataset_name: str) -> list:
        """
        List available splits for a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            List of available splits
        """
        try:
            dataset_info = self.get_dataset_info(dataset_name)
            return list(dataset_info.splits.keys())
        except Exception as e:
            logging.error(f"Error listing splits for {dataset_name}: {str(e)}")
            raise 