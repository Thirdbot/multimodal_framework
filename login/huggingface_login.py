from huggingface_hub import login, HfApi
from datasets import load_dataset
import os
from typing import Optional, Union, Dict, Any
from dotenv import load_dotenv,dotenv_values

load_dotenv()

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
        
        self.api = HfApi()
        self._login()
    
    def _login(self) -> None:
        """Authenticate with Hugging Face Hub."""
        login(token=self.token)
    
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
        return load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
            token=self.token,
            **kwargs
        )
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            Dict containing dataset information
        """
        return self.api.dataset_info(dataset_name)
    
    def list_dataset_splits(self, dataset_name: str) -> list:
        """
        List available splits for a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            List of available splits
        """
        dataset_info = self.get_dataset_info(dataset_name)
        return list(dataset_info.splits.keys()) 