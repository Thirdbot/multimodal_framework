# Multimodal Framework

A framework for building and fine-tuning multimodal (vision-language) models, primarily based on the Qwen architecture, using PyTorch, Hugging Face `transformers`, and `peft`.

This project handles:
- Downloading datasets and models from Hugging Face.
- Preparing and formatting datasets for multimodal training.
- Fine-tuning vision and language models with LoRA/QLoRA.
- Running inference for vision-language tasks.

---

## Prerequisites

- **Python 3.9** or later.
- **CUDA-compatible GPU** (recommended for fine-tuning) or CPU (for light inference).
- **Hugging Face Hub account** with an access token for gated models and datasets.

---

## Installation

### 1. Install PyTorch
Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) to find the correct installation command for your system (Windows/Linux/Mac and CUDA/CPU).

Example for Linux/Windows with CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install Project Dependencies
After installing PyTorch, install the remaining dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
The project uses the following environment variables:
- `hf_token`: Your Hugging Face access token (Required for many models/datasets).
  ```bash
  export hf_token=YOUR_HUGGINGFACE_TOKEN  # On Windows: set hf_token=YOUR_HUGGINGFACE_TOKEN
  ```
- `WORKSPACE_DIR`: (Optional) Custom project root directory.
- `MODEL_DIR`: (Optional) Custom directory for model storage.
- `OFFLOAD_DIR`: (Optional) Custom directory for offloading data.

---

## Project Structure

```text
multimodal_framework/
├── main.py              # Main entry point (Download -> Prepare -> Train -> Inference)
├── createmodel.py       # Script to create/wrap base models with vision/conversation capabilities
├── modules/             # Core logic and classes
│   ├── ApiDump.py       # Hugging Face API interaction helpers
│   ├── DataDownload.py  # Dataset and model download management
│   ├── DataModelPrepare.py # Dataset formatting and preparation logic
│   ├── inference.py     # InferenceManager for vision-language generation
│   ├── train.py         # Fine-tuning logic and Trainer setup
│   ├── variable.py      # Path management and global configurations
│   ├── models/          # Model architecture definitions (Conversation/Vision wrappers)
│   └── ModelUtils.py    # Utilities for loading and saving models
├── configs/             # JSON configuration files
│   ├── ApiCardSet.json  # Tracks models and datasets to download/prepare
│   └── saved_config.json # Stores saved configuration state
├── checkpoints/         # Default directory for training checkpoints
├── custom_models/       # Directory for custom model structures and formatted datasets
├── chat_template/       # Jinja2 templates for chat formatting
└── requirements.txt     # Python dependencies
```

---

## Usage

### 1. Model and Dataset Configuration
The framework uses `configs/ApiCardSet.json` to manage what to download and process. You can update this manually or via `main.py`.

In `main.py`, you can specify the models and datasets:
```python
list_models = api.list_models(model_name='Qwen/Qwen1.5-0.5B-Chat', limit=1, gated=False)
list_datasets = api.list_datasets(dataset_name='waltsun/MOAT', limit=1, gated=False)
```

### 2. Preparing and Training
To start the full pipeline (download, format, and fine-tune), run:
```bash
python main.py
```
*Note: You can comment out specific parts in `main.py` if you only want to run inference or skip dataset preparation after the first run.*

### 3. Creating Custom Model Wrappers
Use `createmodel.py` to wrap a base language model (e.g., Qwen) with specific conversation or vision capabilities:
```bash
python createmodel.py
```

### 4. Inference
The `InferenceManager` in `modules/inference.py` handles vision-language generation. Example usage is provided at the bottom of `main.py`.

---

## Tests

Current testing is focused on manual verification and template application:
- `modules/test_apply_template for_multimodal.py`: Tests the multimodal chat template application.
- **TODO**: Implement a comprehensive test suite (e.g., using `pytest`).

---

## Future Roadmap (TODOs)

The following improvements are planned:
- [ ] Migrate inference and training to use the **Unsloth** library for better performance.
- [ ] Implement automatic model pushing to Hugging Face Hub after training.
- [ ] Transition model architecture definitions to use **Keras** only.
- [ ] Adopt **Unsloth GPT-format** for dataset formatting.
- [ ] Expose more model configuration parameters.
- [ ] Move configuration files from JSON to **INI** format.
- [ ] Dockerize the framework for large-scale GPU cluster deployment.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
