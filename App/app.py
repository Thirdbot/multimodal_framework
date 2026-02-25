import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import os
import sys
import threading
import json
import subprocess
import time
from pathlib import Path
import io
from contextlib import redirect_stdout, redirect_stderr

# Add parent directory to path to allow importing modules
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from modules.inference import InferenceManager
from modules.variable import Variable
from modules.ApiDump import ApiCardSetup
from modules.DataDownload import DataLoader
from modules.DataModelPrepare import Manager as DataManager  # Renamed to avoid conflict with App.Manager
from modules.train import FinetuneModel


# --- Configuration ---
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


# --- Helper Classes & Functions ---

class TextboxStream(io.TextIOBase):
    """A custom stream that writes to a CTkTextbox."""

    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, s):
        self.textbox.after(0, self.insert_text, s)

    def insert_text(self, s):
        self.textbox.configure(state="normal")
        self.textbox.insert("end", s)
        self.textbox.see("end")
        self.textbox.configure(state="disabled")


def download_hub_item(repo_id, is_model=True):
    """Downloads a model or dataset from Hugging Face using the framework modules."""
    vars = Variable()
    api = vars.hf_api
    setcard = ApiCardSetup()
    downloader = DataLoader()

    try:
        list_models = None
        list_datasets = None

        if is_model:
            list_models = api.list_models(model_name=repo_id, limit=1)
            if not list_models:
                print(f"Model '{repo_id}' not found on Hugging Face Hub.")
                return False
        else:
            list_datasets = api.list_datasets(dataset_name=repo_id, limit=1)
            if not list_datasets:
                print(f"Dataset '{repo_id}' not found on Hugging Face Hub.")
                return False

        list_download = setcard.set(list_models, list_datasets)
        downloader.run(list_download)
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False


# --- Views (Content Areas) ---

class DashboardView(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.label = ctk.CTkLabel(self, text="Multimodal Framework Dashboard", font=ctk.CTkFont(size=24, weight="bold"))
        self.label.pack(pady=20)
        self.info = ctk.CTkLabel(self, text="Select a tool from the sidebar to get started.")
        self.info.pack(pady=10)

        # Quick Stats
        self.stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.stats_frame.pack(pady=20, padx=20, fill="x")

        self.vars = Variable()
        datasets_count = len(
            list(self.vars.DATASET_FORMATTED_DIR.glob("*"))) if self.vars.DATASET_FORMATTED_DIR.exists() else 0
        models_count = len(list(self.vars.REGULAR_MODEL_DIR.glob("*"))) + len(
            list(self.vars.VISION_MODEL_DIR.glob("*")))

        ctk.CTkLabel(self.stats_frame, text=f"Local Datasets: {datasets_count}", font=ctk.CTkFont(size=14)).pack(
            side="left", padx=20)
        ctk.CTkLabel(self.stats_frame, text=f"Local Models: {models_count}", font=ctk.CTkFont(size=14)).pack(
            side="left", padx=20)


class DatasetRepositoryView(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.variable = Variable()

        self.title = ctk.CTkLabel(self, text="Dataset Repository", font=ctk.CTkFont(size=20, weight="bold"))
        self.title.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        self.scroll_frame = ctk.CTkScrollableFrame(self, label_text="Formatted Datasets")
        self.scroll_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.grid_rowconfigure(1, weight=1)

        self.refresh_list()

        self.actions_frame = ctk.CTkFrame(self)
        self.actions_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")

        self.btn_refresh = ctk.CTkButton(self.actions_frame, text="Refresh", command=self.refresh_list)
        self.btn_refresh.pack(side="left", padx=10, pady=10)

        self.btn_add = ctk.CTkButton(self.actions_frame, text="Import Dataset")
        self.btn_add.pack(side="left", padx=10, pady=10)

    def refresh_list(self):
        for child in self.scroll_frame.winfo_children():
            child.destroy()

        if self.variable.DATASET_FORMATTED_DIR.exists():
            datasets = [d.name for d in self.variable.DATASET_FORMATTED_DIR.iterdir() if d.is_dir()]
            for ds_name in datasets:
                btn = ctk.CTkButton(self.scroll_frame, text=ds_name,
                                    command=lambda n=ds_name: print(f"Selected Dataset: {n}"),
                                    fg_color="transparent", border_width=1, text_color=("gray10", "#DCE4EE"))
                btn.pack(fill="x", padx=5, pady=5)
        else:
            ctk.CTkLabel(self.scroll_frame, text="No formatted datasets found").pack(pady=10)


class ModelRepositoryView(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.variable = Variable()

        self.title = ctk.CTkLabel(self, text="Model Repository", font=ctk.CTkFont(size=20, weight="bold"))
        self.title.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        self.scroll_frame = ctk.CTkScrollableFrame(self, label_text="Local Models & Checkpoints")
        self.scroll_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.grid_rowconfigure(1, weight=1)

        self.refresh_list()

        self.btn_refresh = ctk.CTkButton(self, text="Refresh List", command=self.refresh_list)
        self.btn_refresh.grid(row=2, column=0, padx=20, pady=20, sticky="w")

    def refresh_list(self):
        for child in self.scroll_frame.winfo_children():
            child.destroy()

        # Gather models from various locations
        model_locations = [
            ("Base Models", self.variable.REGULAR_MODEL_DIR),
            ("Vision Models", self.variable.VISION_MODEL_DIR),
            ("Checkpoints", self.variable.CHECKPOINT_DIR)
        ]

        for label, path in model_locations:
            if not path.exists(): continue

            ctk.CTkLabel(self.scroll_frame, text=label, font=ctk.CTkFont(weight="bold")).pack(fill="x", padx=5,
                                                                                              pady=(10, 5))

            # Sub-directories for checkpoints often have task folders
            if label == "Checkpoints":
                for task_dir in path.iterdir():
                    if task_dir.is_dir():
                        for model_dir in task_dir.iterdir():
                            self._add_model_button(f"{task_dir.name}/{model_dir.name}", model_dir)
            else:
                for model_dir in path.iterdir():
                    if model_dir.is_dir():
                        self._add_model_button(model_dir.name, model_dir)

    def _add_model_button(self, display_name, path):
        btn = ctk.CTkButton(self.scroll_frame, text=display_name,
                            command=lambda p=path: print(f"Selected Model Path: {p}"),
                            fg_color="transparent", border_width=1, text_color=("gray10", "#DCE4EE"))
        btn.pack(fill="x", padx=5, pady=2)


class DownloadHubView(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.title = ctk.CTkLabel(self, text="Hugging Face Hub Downloader", font=ctk.CTkFont(size=20, weight="bold"))
        self.title.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        self.search_frame = ctk.CTkFrame(self)
        self.search_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.entry_search = ctk.CTkEntry(self.search_frame,
                                         placeholder_text="Enter Repo ID (e.g., HuggingFaceTB/SmolLM2-360M)")
        self.entry_search.pack(side="left", fill="x", expand=True, padx=10, pady=10)

        self.type_var = ctk.StringVar(value="Model")
        self.type_menu = ctk.CTkOptionMenu(self.search_frame, values=["Model", "Dataset"], variable=self.type_var,
                                           width=100)
        self.type_menu.pack(side="left", padx=5)

        self.btn_search = ctk.CTkButton(self.search_frame, text="Download", command=self.start_download)
        self.btn_search.pack(side="right", padx=10, pady=10)

        self.log_display = ctk.CTkTextbox(self, font=("Consolas", 10))
        self.log_display.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")
        self.log_display.insert("end", "Enter a Hugging Face repository ID and click Download.\n")
        self.log_display.configure(state="disabled")

    def start_download(self):
        repo_id = self.entry_search.get().strip()
        if not repo_id:
            messagebox.showwarning("Input Error", "Please enter a Repository ID")
            return

        is_model = self.type_var.get() == "Model"
        self.btn_search.configure(state="disabled")
        self.append_log(f"Starting download for {repo_id} ({self.type_var.get()})...\n")

        threading.Thread(target=self.run_download, args=(repo_id, is_model), daemon=True).start()

    def run_download(self, repo_id, is_model):
        stream = TextboxStream(self.log_display)
        with redirect_stdout(stream), redirect_stderr(stream):
            success = download_hub_item(repo_id, is_model)
            if success:
                self.after(0, lambda: self.append_log(f"Successfully downloaded {repo_id}\n"))
                messagebox.showinfo("Success", f"Downloaded {repo_id} successfully.")
            else:
                self.after(0, lambda: self.append_log(f"Failed to download {repo_id}. Check console for details.\n"))
                messagebox.showerror("Error", f"Failed to download {repo_id}.")

            self.after(0, lambda: self.btn_search.configure(state="normal"))

    def append_log(self, text):
        self.log_display.configure(state="normal")
        self.log_display.insert("end", text)
        self.log_display.see("end")
        self.log_display.configure(state="disabled")


class InferenceView(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Chat Area
        self.chat_display = ctk.CTkTextbox(self, state="disabled", font=ctk.CTkFont(size=13))
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="nsew")

        # Image Preview (Right Side)
        self.image_preview_frame = ctk.CTkFrame(self, width=200)
        self.image_preview_frame.grid(row=0, column=2, rowspan=2, padx=(0, 20), pady=20, sticky="nsew")
        self.image_label = ctk.CTkLabel(self.image_preview_frame, text="No Image")
        self.image_label.pack(expand=True, padx=10, pady=10)
        self.current_image_path = None

        # Input Area
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="ew")

        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Type message...")
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.entry.bind("<Return>", lambda e: self.send_message())

        self.btn_image = ctk.CTkButton(self.input_frame, text="📷", width=40, command=self.upload_image)
        self.btn_image.pack(side="left", padx=(0, 10))

        self.btn_send = ctk.CTkButton(self.input_frame, text="Send", width=80, command=self.send_message)
        self.btn_send.pack(side="left")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.current_image_path = file_path
            try:
                img = Image.open(file_path)
                # Maintain aspect ratio for preview
                ratio = min(180 / img.width, 180 / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=new_size)
                self.image_label.configure(image=ctk_img, text="")
            except Exception as e:
                print(f"Error loading image: {e}")

    def send_message(self):
        msg = self.entry.get()
        if not msg and not self.current_image_path: return

        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"User: {msg}\n", "user")
        if self.current_image_path:
            self.chat_display.insert("end", f"[Image Attached: {os.path.basename(self.current_image_path)}]\n")
        self.chat_display.insert("end", "\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")
        self.entry.delete(0, "end")

        threading.Thread(target=self.run_inference, args=(msg,), daemon=True).start()

    def run_inference(self, msg):
        vars = Variable()
        # Find latest checkpoint or model
        checkpoint_dir = vars.CHECKPOINT_DIR / "text-vision-text-generation"
        model_dirs = list(checkpoint_dir.glob("*")) if checkpoint_dir.exists() else []

        if not model_dirs:
            # Try text-generation if vision not found
            checkpoint_dir = vars.CHECKPOINT_DIR / "text-generation"
            model_dirs = list(checkpoint_dir.glob("*")) if checkpoint_dir.exists() else []

        if not model_dirs:
            self.after(0, lambda: self.append_response(
                "System: No fine-tuned models found. Please train a model or specify a path in Configuration."))
            return

        try:
            model_path = str(model_dirs[0])
            manager = InferenceManager(model_path)
            response = manager.generate_response(msg, self.current_image_path)
            self.after(0, lambda: self.append_response(f"Assistant: {response}"))
        except Exception as e:
            self.after(0, lambda: self.append_response(f"Error: {str(e)}"))

    def append_response(self, text):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", text + "\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")


class ConfigView(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.title = ctk.CTkLabel(self, text="Configuration Editor", font=ctk.CTkFont(size=20, weight="bold"))
        self.title.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        # File Selection
        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.grid(row=0, column=0, padx=20, pady=(60, 0), sticky="ew")

        self.file_combo = ctk.CTkComboBox(self.file_frame, values=self.get_config_files(), command=self.load_file)
        self.file_combo.pack(side="left", padx=10, pady=10, fill="x", expand=True)

        self.btn_reload = ctk.CTkButton(self.file_frame, text="Reload", width=80,
                                        command=lambda: self.load_file(self.file_combo.get()))
        self.btn_reload.pack(side="left", padx=10)

        self.btn_save = ctk.CTkButton(self.file_frame, text="Save", width=80, command=self.save_file)
        self.btn_save.pack(side="left", padx=10)

        # Editor Area
        self.editor = ctk.CTkTextbox(self, font=("Consolas", 12))
        self.editor.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

        # Load initial file
        if self.get_config_files():
            self.file_combo.set(self.get_config_files()[0])
            self.load_file(self.file_combo.get())

    def get_config_files(self):
        files = []
        vars = Variable()
        config_dir = vars.WORKSPACE / "configs"
        chat_template_dir = vars.WORKSPACE / "chat_template"

        if config_dir.exists():
            files.extend([str(p) for p in config_dir.glob("*.json")])
        if chat_template_dir.exists():
            files.extend([str(p) for p in chat_template_dir.glob("*.jinja")])

        return files

    def load_file(self, filepath):
        if not filepath or not os.path.exists(filepath): return
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            self.editor.delete("1.0", "end")
            self.editor.insert("1.0", content)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def save_file(self):
        filepath = self.file_combo.get()
        if not filepath: return
        try:
            content = self.editor.get("1.0", "end-1c")
            with open(filepath, 'w') as f:
                f.write(content)
            messagebox.showinfo("Success", "File saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")


class TrainingView(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.title = ctk.CTkLabel(self, text="Model Training", font=ctk.CTkFont(size=20, weight="bold"))
        self.title.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        # Controls
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.btn_start = ctk.CTkButton(self.controls_frame, text="Start Training", command=self.start_training,
                                       fg_color="green")
        self.btn_start.pack(side="left", padx=10, pady=10)

        self.btn_stop = ctk.CTkButton(self.controls_frame, text="Stop", command=self.stop_training, fg_color="red",
                                      state="disabled")
        self.btn_stop.pack(side="left", padx=10, pady=10)

        # Logs
        self.log_display = ctk.CTkTextbox(self, font=("Consolas", 10))
        self.log_display.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")
        self.log_display.insert("end", "Ready to start training...\n")
        self.log_display.configure(state="disabled")
        self.training_process = None

    def start_training(self):
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.log_display.configure(state="normal")
        self.log_display.delete("1.0", "end")
        self.log_display.insert("end", "Starting training process...\n")
        self.log_display.configure(state="disabled")

        # Run training in a separate thread
        threading.Thread(target=self.run_training_process, daemon=True).start()

    def stop_training(self):
        if self.training_process and self.training_process.poll() is None:
            self.training_process.terminate()
            self.append_log("Training process terminated by user.")
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def run_training_process(self):
        # This would call your actual training script
        # For now, we simulate output
        cmd = f"python {parent_dir}/main.py"  # Or specific training command

        self.training_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True
        )

        while True:
            line = self.training_process.stdout.readline()
            if not line and self.training_process.poll() is not None:
                break
            if line:
                self.after(0, lambda l=line: self.append_log(l))

        self.after(0, lambda: self.btn_start.configure(state="normal"))
        self.after(0, lambda: self.btn_stop.configure(state="disabled"))
        self.after(0, lambda: self.append_log("Training process finished."))
        self.training_process = None

    def append_log(self, text):
        self.log_display.configure(state="normal")
        self.log_display.insert("end", text)
        self.log_display.see("end")
        self.log_display.configure(state="disabled")


class FormattingView(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.title = ctk.CTkLabel(self, text="Dataset Formatting", font=ctk.CTkFont(size=20, weight="bold"))
        self.title.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        # Controls
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.btn_run = ctk.CTkButton(self.controls_frame, text="Run Formatting", command=self.run_formatting)
        self.btn_run.pack(side="left", padx=10, pady=10)

        # Logs
        self.log_display = ctk.CTkTextbox(self, font=("Consolas", 10))
        self.log_display.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")
        self.log_display.insert("end", "Ready to format datasets...\n")
        self.log_display.configure(state="disabled")

    def run_formatting(self):
        self.btn_run.configure(state="disabled")
        threading.Thread(target=self.run_process, daemon=True).start()

    def run_process(self):
        # Call the formatting logic
        # Assuming main.py or a specific module handles this
        cmd = f"python {parent_dir}/modules/DataModelPrepare.py"

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True
        )

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                self.after(0, lambda l=line: self.append_log(l))

        self.after(0, lambda: self.btn_run.configure(state="normal"))
        self.after(0, lambda: self.append_log("Formatting finished."))

    def append_log(self, text):
        self.log_display.configure(state="normal")
        self.log_display.insert("end", text)
        self.log_display.see("end")
        self.log_display.configure(state="disabled")


class WorkflowView(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)  # Log display takes most space
        self.variable = Variable()

        self.title = ctk.CTkLabel(self, text="Training Workflow", font=ctk.CTkFont(size=20, weight="bold"))
        self.title.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        # Selection Frame
        self.selection_frame = ctk.CTkFrame(self)
        self.selection_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.selection_frame.grid_columnconfigure(0, weight=1)
        self.selection_frame.grid_columnconfigure(1, weight=1)

        # Model Selection
        ctk.CTkLabel(self.selection_frame, text="Select Model:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.model_options = self._get_local_models()
        self.model_var = ctk.StringVar(value=self.model_options[0] if self.model_options else "No Models Found")
        self.model_selector = ctk.CTkComboBox(self.selection_frame, values=self.model_options, variable=self.model_var)
        self.model_selector.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Dataset Selection
        ctk.CTkLabel(self.selection_frame, text="Select Datasets:").grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.dataset_checkboxes = []
        self.dataset_vars = {}
        self.dataset_scroll_frame = ctk.CTkScrollableFrame(self.selection_frame, height=100)
        self.dataset_scroll_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
        self._populate_datasets()

        # Action Buttons
        self.action_frame = ctk.CTkFrame(self)
        self.action_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.btn_start_workflow = ctk.CTkButton(self.action_frame, text="Start Workflow", command=self.start_workflow,
                                                fg_color="green")
        self.btn_start_workflow.pack(side="left", padx=10, pady=10)

        self.btn_refresh_workflow = ctk.CTkButton(self.action_frame, text="Refresh Lists", command=self.refresh_lists)
        self.btn_refresh_workflow.pack(side="left", padx=10, pady=10)

        # Log Display
        self.log_display = ctk.CTkTextbox(self, font=("Consolas", 10))
        self.log_display.grid(row=3, column=0, padx=20, pady=20, sticky="nsew")
        self.log_display.insert("end", "Select a model and one or more datasets to start the workflow.\n")
        self.log_display.configure(state="disabled")

    def _get_local_models(self):
        models = []
        # Gather models from various locations
        model_locations = [
            self.variable.REGULAR_MODEL_DIR,
            self.variable.VISION_MODEL_DIR,
            self.variable.CHECKPOINT_DIR,
            self.variable.LocalModel_DIR
        ]
        for path in model_locations:
            if path.exists():
                for item in path.iterdir():
                    # owner name
                    if item.is_dir():
                        for sub_item in item.iterdir():
                            # model actual name
                            if sub_item.is_dir():
                                models.append(f"{item.name}/{sub_item.name}")

        return sorted(list(set(models)))  # Remove duplicates and sort

    def _populate_datasets(self):
        for widget in self.dataset_scroll_frame.winfo_children():
            widget.destroy()
        self.dataset_checkboxes = []
        self.dataset_vars = {}

        if self.variable.DATASETS_DIR.exists():
            dir_listed = []
            #sub de
            datasets = [ d.name for d in self.variable.DATASETS_DIR.iterdir() if d.is_dir()]

            for ds_name in sorted(datasets):
                var = ctk.BooleanVar(value=False)
                chk = ctk.CTkCheckBox(self.dataset_scroll_frame, text=ds_name, variable=var)
                chk.pack(anchor="w", padx=5, pady=2)
                self.dataset_checkboxes.append(chk)
                self.dataset_vars[ds_name] = var
        else:
            ctk.CTkLabel(self.dataset_scroll_frame, text="No formatted datasets found").pack(pady=10)

    def refresh_lists(self):
        self.model_options = self._get_local_models()
        self.model_selector.configure(values=self.model_options)
        if self.model_options:
            self.model_var.set(self.model_options[0])
        else:
            self.model_var.set("No Models Found")
        self._populate_datasets()
        self.append_log("Model and Dataset lists refreshed.\n")

    def start_workflow(self):
        selected_model = self.model_var.get()
        selected_datasets = [name for name, var in self.dataset_vars.items() if var.get()]

        if selected_model == "No Models Found" or not selected_datasets:
            messagebox.showwarning("Selection Error", "Please select a model and at least one dataset.")
            return

        self.btn_start_workflow.configure(state="disabled")
        self.btn_refresh_workflow.configure(state="disabled")
        self.append_log(f"Starting workflow with Model: {selected_model}, Datasets: {', '.join(selected_datasets)}\n")

        threading.Thread(target=self._run_workflow_process, args=(selected_model, selected_datasets),
                         daemon=True).start()

    def _run_workflow_process(self, model_name, dataset_names):
        stream = TextboxStream(self.log_display)
        with redirect_stdout(stream), redirect_stderr(stream):
            try:

                # 1. Prepare list_download for ApiCardSetup
                list_download_dict = {'model': {}}
                list_download_dict['model'][model_name] = {ds_name: "" for ds_name in dataset_names}

                print(f"\n--- Step 1: Updating ApiCardSet.json ---")
                set_card = ApiCardSetup()
                # 2. set model and dataset queue
                list_data = set_card.set(from_repository=list_download_dict)

                # 3. Run DataModelPrepare (formatting)
                print(f"\n--- Step 2: Formatting Datasets ---")
                data_manager = DataManager()
                data_manager.dataset_prepare(list_data)

                # 4. Run FinetuneModel (training)
                print(f"\n--- Step 4: Starting Fine-tuning ---")
                finetuner = FinetuneModel()
                finetuner.finetune_model()

                print(f"\n--- Workflow Completed Successfully! ---")
                messagebox.showinfo("Workflow Complete", "Training workflow finished successfully!")

            except Exception as e:
                print(f"\n--- Workflow Failed! ---")
                print(f"An error occurred during workflow: {e}")
                messagebox.showerror("Workflow Error", f"An error occurred during workflow: {e}")
            finally:
                self.after(0, lambda: self.btn_start_workflow.configure(state="normal"))
                self.after(0, lambda: self.btn_refresh_workflow.configure(state="normal"))

    def append_log(self, text):
        self.log_display.configure(state="normal")
        self.log_display.insert("end", text)
        self.log_display.see("end")
        self.log_display.configure(state="disabled")


class TopToolbar(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, height=40, corner_radius=0, **kwargs)
        self.pack_propagate(False)  # Fixed height

        self.btn_file = ctk.CTkButton(self, text="File", width=60, fg_color="transparent",
                                      text_color=("gray10", "#DCE4EE"), hover_color=("gray70", "gray30"))
        self.btn_file.pack(side="left", padx=5)

        self.btn_view = ctk.CTkButton(self, text="View", width=60, fg_color="transparent",
                                      text_color=("gray10", "#DCE4EE"), hover_color=("gray70", "gray30"))
        self.btn_view.pack(side="left", padx=5)

        self.btn_help = ctk.CTkButton(self, text="Help", width=60, fg_color="transparent",
                                      text_color=("gray10", "#DCE4EE"), hover_color=("gray70", "gray30"))
        self.btn_help.pack(side="left", padx=5)

        self.lbl_status = ctk.CTkLabel(self, text="Ready", text_color="gray")
        self.lbl_status.pack(side="right", padx=20)


class SidebarNavigation(ctk.CTkFrame):
    def __init__(self, master, command=None, **kwargs):
        super().__init__(master, width=200, corner_radius=0, **kwargs)
        self.command = command
        self.grid_rowconfigure(10, weight=1)  # Spacer at bottom

        self.logo = ctk.CTkLabel(self, text="Multimodal\nFramework", font=ctk.CTkFont(size=18, weight="bold"))
        self.logo.grid(row=0, column=0, padx=20, pady=20)

        self.buttons = []

        # Navigation Items
        items = [
            ("Dashboard", "dashboard"),
            ("Dataset Repo", "dataset_repo"),
            ("Model Repo", "model_repo"),
            ("Formatting", "formatting"),
            ("Hubs / Download", "hubs"),
            ("Workflow", "workflow"),  # New Workflow button
            ("Training", "training"),  # Keep for direct training if needed
            ("Inference", "inference"),
            ("Configuration", "config")
        ]

        for i, (text, name) in enumerate(items):
            btn = ctk.CTkButton(self, text=text, anchor="w",
                                command=lambda n=name: self.button_click(n),
                                fg_color="transparent", text_color=("gray10", "#DCE4EE"))
            btn.grid(row=i + 1, column=0, sticky="ew", padx=10, pady=2)
            self.buttons.append(btn)

        # Appearance Mode at bottom
        self.appearance_menu = ctk.CTkOptionMenu(self, values=["System", "Light", "Dark"],
                                                 command=self.change_appearance)
        self.appearance_menu.grid(row=11, column=0, padx=20, pady=20, sticky="s")

    def button_click(self, name):
        if self.command:
            self.command(name)

    def change_appearance(self, mode):
        ctk.set_appearance_mode(mode)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Multimodal Framework Studio")
        self.geometry("1280x720")

        # Layout Grid
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # 1. Top Toolbar
        self.toolbar = TopToolbar(self)
        self.toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")

        # 2. Sidebar
        self.sidebar = SidebarNavigation(self, command=self.show_view)
        self.sidebar.grid(row=1, column=0, sticky="nsew")

        # 3. Content Area Container
        self.content_area = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.content_area.grid(row=1, column=1, sticky="nsew")
        self.content_area.grid_rowconfigure(0, weight=1)
        self.content_area.grid_columnconfigure(0, weight=1)

        # Initialize Views
        self.views = {
            "dashboard": DashboardView(self.content_area),
            "dataset_repo": DatasetRepositoryView(self.content_area),
            "model_repo": ModelRepositoryView(self.content_area),
            "formatting": FormattingView(self.content_area),
            "hubs": DownloadHubView(self.content_area),
            "workflow": WorkflowView(self.content_area),  # New Workflow View
            "training": TrainingView(self.content_area),
            "inference": InferenceView(self.content_area),
            "config": ConfigView(self.content_area)
        }

        # Add placeholders text for views without custom classes yet
        for name, view in self.views.items():
            if len(view.winfo_children()) == 0:
                ctk.CTkLabel(view, text=f"{name.replace('_', ' ').title()} View - Under Construction").pack(expand=True)

        # Show default view
        self.show_view("dashboard")

    def show_view(self, name):
        # Hide all views
        for view in self.views.values():
            view.grid_forget()

        # Show selected view
        if name in self.views:
            self.views[name].grid(row=0, column=0, sticky="nsew")
            if hasattr(self.views[name], "refresh_list"):
                self.views[name].refresh_list()
            if hasattr(self.views[name], "refresh_lists"):  # For WorkflowView
                self.views[name].refresh_lists()


if __name__ == "__main__":
    app = App()
    app.mainloop()