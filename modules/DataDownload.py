import json
from dataclasses import dataclass
from pathlib import Path

from colorama import Fore, Style, init
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModel, AutoTokenizer

from modules.variable import Variable

init(autoreset=True)


@dataclass
class DatasetDownloadConfig:
    allow_download: tuple[str, ...] = ("*.json", "*.csv", "*.parquet", "*.zip")
    default_split: str = "train"
    saved_config_file: Path | None = None
    datasets_dir: Path | None = None
    repo_dir: Path | None = None


@dataclass
class ModelDownloadConfig:
    local_model_dir: Path | None = None

"""Download locally as separated folders for training and inference."""


class FlexibleDatasetLoader:
    def __init__(self, split: str = "train", config: DatasetDownloadConfig | None = None):
        self.variable = Variable()
        self.split = split
        self.dataset = None
        self.config = config or DatasetDownloadConfig()
        self.saved_configs: dict[str, str | None] = {}

        # Resolve paths from config or defaults
        self.DATASETS_DIR = self.config.datasets_dir or self.variable.DATASETS_DIR
        self.SAVED_CONFIG_FILE = self.config.saved_config_file or self.variable.SAVED_CONFIG_Path
        # Ensure target directories exist
        self.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        self.saved_configs = self._load_saved_configs()

    def load(self, name: str, config_name: str | None):
        """Load a dataset and download its artifacts locally."""
        print('name:', name)
        print('config:', config_name)

        self.split = self._select_split(name, config_name)

        if config_name:
            print(f"load with config {config_name}")
            try:
                dataset_dir = self._dataset_dir(name)

                self.dataset = load_dataset(
                    name,
                    config_name,
                    split=self.split,
                )
                print(f"{Fore.GREEN}Successfully loaded dataset {name}{Style.RESET_ALL}")

            except Exception as e:
                print(f"{e}")
                return self.load(name, None)

        else:
            configs = get_dataset_config_names(name)
            user_input = None
            print(f"{Fore.CYAN}Available configs for {name}: {configs}{Style.RESET_ALL}")
            saved_config = self.saved_configs
            if name in saved_config:
                user_input = saved_config[name]
            else:
                user_input = configs[0] if configs else None
                saved_config[name] = user_input
                self.saved_configs = saved_config
                with open(self.SAVED_CONFIG_FILE, 'w') as f:
                    json.dump(saved_config, f, indent=4)
            return self.load(name, user_input)

        try:
            snapshot_download(
                repo_id=name,
                revision='main',
                local_dir=self._dataset_dir(name),
                allow_patterns=self.config.allow_download,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"{Fore.RED}Error downloading dataset {name}: {str(e)}{Style.RESET_ALL}")

    def get(self):
        return self.dataset

    def _select_split(self, name: str, config_name: str | None) -> str:
        try:
            splits = get_dataset_split_names(name, config_name)
            if 'train' in splits:
                return 'train'
            if 'test' in splits:
                return 'test'
            return self.config.default_split
        except Exception:
            return self.config.default_split

    def _load_saved_configs(self) -> dict[str, str | None]:
        try:
            if self.SAVED_CONFIG_FILE.exists():
                with open(self.SAVED_CONFIG_FILE, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _dataset_dir(self, name: str) -> Path:
        short_name = name.split('/')[-1]
        dataset_dir = self.DATASETS_DIR / short_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir



class ModelLoader:
    def __init__(self, config: ModelDownloadConfig | None = None):
        self.variable = Variable()
        self.config = config or ModelDownloadConfig()

        self.SAVEDMODEL_DIR = self.config.local_model_dir or self.variable.LocalModel_DIR
        self.SAVEDMODEL_DIR.mkdir(parents=True, exist_ok=True)

    def load_model(self, name: str):
        try:
            local_path = Path(name)
            if local_path.exists():
                print(f"{Fore.GREEN}Using custom model from: {local_path}{Style.RESET_ALL}")
                return local_path

            model_dir = self.SAVEDMODEL_DIR / name
            if model_dir.exists():
                print(f"{Fore.GREEN}Model already cached at: {model_dir}{Style.RESET_ALL}")
                return model_dir

            model = AutoModel.from_pretrained(name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"{Fore.GREEN}Saved model and tokenizer to: {model_dir}{Style.RESET_ALL}")
            return model_dir
        except Exception as e:
            print(f"{Fore.RED}Error downloading model {name}: {str(e)}{Style.RESET_ALL}")
            return None


class DataLoader:
    def __init__(self,
                 dataset_config: DatasetDownloadConfig | None = None,
                 model_config: ModelDownloadConfig | None = None):

        self.model = ModelLoader(config=model_config)
        self.dataset = FlexibleDatasetLoader(config=dataset_config)

    def run(self, params):
        return self.load(params)

    def load(self, to_install):
        for model, datasets in to_install['model'].items():
            print('downloading model:', model)
            print('downloading dataset:', datasets)

            try:
                self.model.load_model(model)
                print(f"{Fore.CYAN}Processing model: {model} with datasets: {datasets}{Style.RESET_ALL}")

                if isinstance(datasets, dict):
                    try:
                        for dataset, info in datasets.items():
                            dataset_config_name = None
                            if isinstance(self.dataset.saved_configs, dict):
                                dataset_config_name = self.dataset.saved_configs.get(dataset)
                            self.dataset.load(dataset, dataset_config_name)
                    except Exception as e:
                        print(f"{Fore.RED}Error processing dataset {dataset}: {str(e)}{Style.RESET_ALL}")
                    
            except Exception as e:
                print(f"{Fore.RED}Error processing model {model}: {str(e)}{Style.RESET_ALL}")
    
   
            
    