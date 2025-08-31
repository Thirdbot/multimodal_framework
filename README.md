# Multimodal Framework

This project is a multimodal framework designed to handle datasets, fine-tune models, and integrate vision and language models for multimodal tasks. It supports downloading datasets and models from Hugging Face, preparing datasets for fine-tuning, and managing the training process.

---

## Prerequisites

- Python 3.9 or later
- pip (Python package installer)
- Hugging Face Hub account (with an access token)
- PyTorch and related dependencies

---

## Installation

### 1. Install PyTorch

Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) and follow these steps:

1. Select your preferences:
   - PyTorch Build: Stable (2.1.0)
   - Your OS: Windows
   - Package: Pip
   - Language: Python
   - Compute Platform: CUDA 11.8 (if you have an NVIDIA GPU) or CPU

2. Run the generated command in your terminal. For example:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # For CPU only
   pip install torch torchvision torchaudio
   ```

### 2. Install Project Dependencies

After installing PyTorch, install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

### 3. Set Your Hugging Face Access Token

Set your Hugging Face access token as an environment variable:
```bash
export hf_token=YOUR_HUGGINGFACE_TOKEN  # On Windows: set hf_token=YOUR_HUGGINGFACE_TOKEN
```

---

## Project Structure

```
multimodal_framework/
├── main.py              # Main application entry point
├── modules/             # Core modules for the framework
│   ├── ApiDump.py       # Handles Hugging Face API interactions
│   ├── DataDownload.py  # Manages dataset and model downloading
│   ├── DataModelPrepare.py  # Prepares datasets and manages fine-tuning
│   ├── variable.py      # Stores global variables and configurations
│   └── createbasemodel.py  # Defines the multimodal model architecture
├── models/              # Model storage directory
├── offload/             # Offloading directory
├── login/               # Login-related modules
└── README.md            # Documentation for the project
```

---

## Environment Setup

The project uses the following environment variables:
- `WORKSPACE_DIR`: Project root directory
- `MODEL_DIR`: Directory for model storage
- `OFFLOAD_DIR`: Directory for offloading data

---

## Usage

### **1. Prepare the Dataset and Model**
The script automatically downloads the specified dataset and model from Hugging Face. You can modify the dataset and model in the `main.py` file:
```python
list_models = api.list_models(model_name='Qwen/Qwen1.5-0.5B-Chat', limit=1, gated=False)
list_datasets = api.list_datasets(dataset_name='waltsun/MOAT', limit=1, gated=False)
```

### **2. Run the Script**
To start the process, simply run:
```bash
python main.py
```

### **3. What Happens During Execution**
- **Dataset and Model Download:**  
  The script downloads the specified dataset and model using the Hugging Face API.
- **Dataset Preparation:**  
  The `Manager` class prepares the dataset for multimodal fine-tuning.
- **Fine-Tuning:**  
  The script fine-tunes the model using the prepared dataset.

---

## Notes

- Make sure you have sufficient disk space for model downloads.
- GPU acceleration is recommended for better performance.
- The project uses Hugging Face models, so ensure you have proper authentication set up.

---

## Troubleshooting

If you encounter any issues during installation or execution:

1. Verify your Python version:
   ```bash
   python --version
   ```
2. Ensure pip is up to date:
   ```bash
   pip install --upgrade pip
   ```
3. Check CUDA installation (if using GPU):
   ```bash
   nvidia-smi
   ```
4. Clear pip cache if needed:
   ```bash
   pip cache purge
   ```

---

## Future Work

- Add support for VLLM inference.
- Dockerize the framework for deployment on large GPU clusters.
- Implement RunPod for managing training jobs.
- Improve dataset merging and rearrangement for multimodal training.

---

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
