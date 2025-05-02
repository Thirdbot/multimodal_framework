# Multimodal Agentic Project

This project requires PyTorch and related dependencies. Please follow the instructions below to set up your environment.

## Prerequisites

- Python 3.9 or later
- pip (Python package installer)

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
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # For CPU only
   pip3 install torch torchvision torchaudio
   ```

### 2. Install Project Dependencies

After installing PyTorch, install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
multimodal_agentic/
├── main.py              # Main application entry point
├── models/              # Model definitions
│   └── Text-Text-generation/  # Model storage
├── offload/             # Offloading directory
└── login/              # Login related modules
```

## Environment Setup

The project uses the following environment variables:
- `WORKSPACE_DIR`: Project root directory
- `MODEL_DIR`: Directory for model storage
- `OFFLOAD_DIR`: Directory for offloading data

## Usage

1. Ensure all dependencies are installed
2. Run the main application:
   ```bash
   python main.py
   ```

## Notes

- Make sure you have sufficient disk space for model downloads
- GPU acceleration is recommended for better performance
- The project uses Hugging Face models, so ensure you have proper authentication set up

## Troubleshooting

If you encounter any issues during installation:

1. Verify your Python version: `python --version`
2. Ensure pip is up to date: `pip install --upgrade pip`
3. Check CUDA installation (if using GPU): `nvidia-smi`
4. Clear pip cache if needed: `pip cache purge`
