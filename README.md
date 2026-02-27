# Model DEV Studio

A local fine-tuning framework for HuggingFace language models with support for both **text (conversation)** and **vision** modalities. Provides a full pipeline from downloading base models to running inference on trained checkpoints, accessible via either a graphical interface or the command line.

---

## Features

- Download models and datasets directly from HuggingFace Hub
- Wrap base models with LoRA adapters for efficient fine-tuning
- Support for **conversation models** and **multimodal vision models** (CLIP + LM)
- Chat-template-based dataset formatting and tokenization
- HuggingFace `Trainer`-based fine-tuning with GPU/CPU support
- Inference on saved checkpoints (text-only or image + text)
- GUI built with CustomTkinter and CLI entry points for every module

---

## Prerequisites

- **Python 3.9** or later
- **CUDA-compatible GPU** (recommended for fine-tuning) or CPU (for light inference)
- **HuggingFace Hub account** with an access token for gated models/datasets

---

## Installation

### 1. Install PyTorch

Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for the correct command for your system.

Example for Linux/Windows with CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install project dependencies

```bash
pip install -r requirements.txt
```

### 3. HuggingFace token (optional, required for gated repos)

```bash
export hf_token=YOUR_HUGGINGFACE_TOKEN   # Linux/macOS
set hf_token=YOUR_HUGGINGFACE_TOKEN      # Windows
```

---

## Project Structure

```
multimodal_framework/
├── App/
│   ├── app.py              # GUI application entry point
│   └── ui_components.py    # CustomTkinter UI components
├── modules/
│   ├── variable.py         # Central path/config object
│   ├── ApiDump.py          # ApiCard manager (model-dataset registry)
│   ├── DataDownload.py     # HuggingFace Hub downloader
│   ├── DataModelPrepare.py # Dataset formatting and tokenization
│   ├── ModelUtils.py       # Model wrappers, LoRA helpers, load/save
│   ├── train.py            # Fine-tuning pipeline
│   ├── inference.py        # Inference manager
│   ├── chatTemplate.py     # Chat template formatter
│   ├── prerun.py           # Workspace initializer
│   └── models/             # Custom model config classes
├── configs/                # ApiCardSet.json, saved_config.json
├── repositories/           # Downloaded models and datasets
│   ├── models/
│   └── datasets/
├── custom_models/          # LoRA-wrapped models ready for training
│   ├── conversation-model/
│   └── vision-model/
├── formatted_datasets/     # Tokenized datasets saved to disk
├── checkpoints/            # Training output checkpoints
├── chat_template/          # Jinja2 chat template files
├── createmodel.py          # CLI script to wrap a base model
├── main.py                 # CLI pipeline reference script
└── requirements.txt
```

---

## GUI

Launch the graphical interface:

```bash
python App/app.py
```

**Model DEV Studio** opens with a sidebar navigation. All long-running operations execute in background threads so the interface stays responsive. A **global terminal** at the bottom streams live log output from every operation.

### Views

| View | Description |
|------|-------------|
| **Download** | Enter a HuggingFace repo ID and download a model or dataset |
| **Formatted Dataset** | Browse tokenized datasets ready for training |
| **Custom Model** | Browse LoRA-wrapped models available for training or formatting |
| **Train** | Start fine-tuning; live logs stream to the view and the global terminal |
| **Create Model** | Wrap a downloaded base model with LoRA adapters |
| **Format Dataset and Set Task** | Apply chat templates and tokenize datasets |
| **Configuration** | View and edit JSON config files |

---

## CLI

Every module has a CLI entry point. The sections below cover each step of the pipeline.

---

## Workflow

### Step 1 — Download a model or dataset

**GUI:** Go to **Download**, enter a repo ID (e.g. `HuggingFaceTB/SmolLM2-360M`), select *Model* or *Dataset*, click **Download**.

**CLI:**
```bash
# Download a model
python -m modules.DataDownload HuggingFaceTB/SmolLM2-360M

# Download a dataset
python -m modules.DataDownload Lin-Chen/ShareGPT4V --dataset
```

Files are saved to `repositories/models/` and `repositories/datasets/`.

---

### Step 2 — Create a custom model (wrap with LoRA)

Wraps a downloaded base model with LoRA adapters and saves it under `custom_models/`.

**GUI:** Go to **Create Model**, select a model, choose *Conversation* or *Vision* mode, click **Create Wrapped Model**.

**CLI:** Edit the paths in `createmodel.py` then run:
```bash
python createmodel.py
```

**API:**
```python
from modules.ModelUtils import CreateModel

# Conversation model
creator = CreateModel("repositories/models/Qwen/Qwen1.5-0.5B-Chat", "conversation-model")
creator.add_conversation()
creator.save_regular_model()

# Vision model
creator = CreateModel("repositories/models/Qwen/Qwen1.5-0.5B-Chat", "vision-model")
creator.add_vision()
creator.save_vision_model()
```

---

### Step 3 — Format dataset for training

Applies a chat template and tokenizes the dataset, saving the result to `formatted_datasets/`. Also writes `custom_models/training_config.json`, which links model names to dataset column metadata used by the trainer.

**GUI:** Go to **Format Dataset and Set Task**, select a custom model and one or more datasets, click **Run Formatting Process**.

**CLI:**
```bash
python -m modules.DataModelPrepare
```

**API:**
```python
from modules.DataModelPrepare import Manager

api_card = {
    "model": {
        "conversation-model/Qwen/Qwen1.5-0.5B-Chat": {
            "ShareGPT4V": ""
        }
    }
}
Manager().dataset_prepare(api_card)
```

---

### Step 4 — Fine-tune

Reads `training_config.json`, loads each custom model and its formatted dataset, and runs HuggingFace `Trainer`. Checkpoints are saved under `checkpoints/`.

**GUI:** Go to **Train**, click **Start Training**.

**CLI:**
```bash
python -m modules.train
```

**API:**
```python
from modules.train import FinetuneModel
FinetuneModel().finetune_model()
```

Training hyperparameters (batch size, learning rate, epochs) can be adjusted in `modules/train.py` inside `FinetuneModel.__init__`.

---

### Step 5 — Run inference

**CLI:**
```bash
# Text-only
python -m modules.inference \
    --model checkpoints/text-generation/Qwen_Qwen1.5-0.5B-Chat \
    --prompt "Explain transformers in simple terms"

# With an image (vision model)
python -m modules.inference \
    --model checkpoints/text-vision-text-generation/Qwen_Qwen1.5-0.5B-Chat \
    --prompt "Describe this image" \
    --image /path/to/image.jpg
```

**API:**
```python
from modules.inference import InferenceManager

manager = InferenceManager("checkpoints/text-generation/Qwen_Qwen1.5-0.5B-Chat")
print(manager.generate_response("Hello, how are you?"))

# Vision inference
print(manager.generate_response("What is in this image?", image_path="image.jpg"))
```

---

## Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| `ApiCardSet.json` | `configs/` | Registry linking model names to dataset names |
| `saved_config.json` | `configs/` | Saved HuggingFace dataset config choices |
| `training_config.json` | `custom_models/` | Generated after formatting; used by the trainer |

---

## Supported Architectures (LoRA)

Pre-configured LoRA target modules are included for:

`gpt2` · `llama` · `mistral` · `opt` · `bloom` · `t5` · `bert` · `roberta` · `gpt_neox` · `falcon` · `mpt` · `baichuan` · `chatglm` · `qwen` · `phi` · `gemma` · `stablelm`

For any other architecture, LoRA is skipped and the full model is fine-tuned.

---

## Future Roadmap

- [ ] Migrate inference and training to the **Unsloth** library for better performance
- [ ] Push trained models to HuggingFace Hub after training
- [ ] Adopt **Unsloth GPT-format** for dataset formatting
- [ ] Expose more model configuration parameters
- [ ] Move configuration files from JSON to INI format

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.