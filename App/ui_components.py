import io
import threading
from contextlib import redirect_stdout

import customtkinter as ctk
import os
from tkinter import messagebox


class LoggerFrame(ctk.CTkFrame):
    """Generic frame with a title, a button, and a log output."""

    def __init__(self, master, title, button_text, run_command, **kwargs):
        super().__init__(master, **kwargs)
        ctk.CTkLabel(self, text=title, font=("", 20, "bold")).pack(pady=10, padx=20, anchor="w")

        self.btn = ctk.CTkButton(self, text=button_text, command=lambda: run_command(self))
        self.btn.pack(pady=10, padx=20, anchor="w")

        self.log = ctk.CTkTextbox(self, font=("Consolas", 11))
        self.log.pack(fill="both", expand=True, padx=20, pady=20)

    def write_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")


class ListFrame(ctk.CTkFrame):
    """Generic frame to display a list of items (Models or Datasets)."""

    def __init__(self, master, title, get_items_func,get_pass_in_func=None, **kwargs):
        super().__init__(master, **kwargs)
        ctk.CTkLabel(self, text=title, font=("", 20, "bold")).pack(pady=10, padx=20, anchor="w")

        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.pack(fill="both", expand=True, padx=20, pady=10)

        self.refresh_btn = ctk.CTkButton(self, text="Refresh", command=self.update_list)
        self.refresh_btn.pack(pady=10, padx=20, anchor="w")
        self.get_items = get_items_func
        self.update_list()

    def update_list(self):
        for child in self.scroll.winfo_children(): child.destroy()
        for item in self.get_items():
            ctk.CTkButton(self.scroll, text=item, fg_color="transparent", border_width=1, anchor="w").pack(fill="x",
                                                                                                           pady=2)

class DownloadHubView(ctk.CTkFrame):
    def __init__(self, master, start_download_callback, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text="Hugging Face Hub Downloader", font=("", 20, "bold")).pack(pady=20, padx=20, anchor="w")

        # Input Area
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.pack(fill="x", padx=20, pady=10)

        self.repo_entry = ctk.CTkEntry(self.input_frame, placeholder_text="Repo ID (e.g., google/gemma-2b)")
        self.repo_entry.pack(side="left", fill="x", expand=True, padx=10, pady=10)

        self.type_var = ctk.StringVar(value="Model")
        self.type_menu = ctk.CTkOptionMenu(self.input_frame, values=["Model", "Dataset"], variable=self.type_var, width=100)
        self.type_menu.pack(side="left", padx=5)

        self.dl_btn = ctk.CTkButton(self.input_frame, text="Download",
                                     command=lambda: start_download_callback(self))
        self.dl_btn.pack(side="right", padx=10)

        ctk.CTkLabel(self, text="Tip: Check the global terminal below for progress.", text_color="gray").pack(padx=20, anchor="w")


import customtkinter as ctk


class TaskSetterView(ctk.CTkFrame):
    def __init__(self, master, vars, on_set_callback, **kwargs):
        super().__init__(master, **kwargs)
        self.vars = vars
        self.on_set_callback = on_set_callback

        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self, text="Set Training Task (Update ApiCard)", font=("", 20, "bold")).grid(row=0, column=0, columnspan=2, pady=10, padx=20, sticky="w")

        # 1. Model Selection (Single)
        self.model_frame = ctk.CTkScrollableFrame(self, label_text="Select Base Model")
        self.model_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.model_var = ctk.StringVar()

        # 2. Dataset Selection (Multiple)
        self.dataset_frame = ctk.CTkScrollableFrame(self, label_text="Select Datasets")
        self.dataset_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.dataset_vars = {}

        # 3. Action Button
        self.set_btn = ctk.CTkButton(self, text="Save Task to ApiCard", command=self.submit_task, fg_color="#2c6e49")
        self.set_btn.grid(row=2, column=0, columnspan=2, pady=20, padx=20, sticky="ew")

        self.refresh_selectors()

    def refresh_selectors(self):
        # Clear old
        for child in self.model_frame.winfo_children(): child.destroy()
        for child in self.dataset_frame.winfo_children(): child.destroy()

        # Populate Models
        model_paths = [self.vars.CUSTOM_MODEL_DIR,self.vars.LocalModel_DIR,self.vars.CHECKPOINT_DIR]
        for p in model_paths:
            if p.exists():
                for d in p.iterdir():
                    if d.is_dir():
                        for sub in d.iterdir():  # Handle owner/model structure
                            if sub.is_dir():
                                name = f"{d.name}/{sub.name}"
                                ctk.CTkRadioButton(self.model_frame, text=name, variable=self.model_var,
                                                   value=name).pack(anchor="w", pady=2)

        # Populate Datasets
        if self.vars.DATASET_FORMATTED_DIR.exists():
            for d in self.vars.DATASET_FORMATTED_DIR.iterdir():
                if d.is_dir():
                    var = ctk.BooleanVar(value=False)
                    self.dataset_vars[d.name] = var
                    ctk.CTkCheckBox(self.dataset_frame, text=d.name, variable=var).pack(anchor="w", pady=2)

    def submit_task(self):
        selected_model = self.model_var.get()
        selected_datasets = [name for name, var in self.dataset_vars.items() if var.get()]

        if not selected_model or not selected_datasets:
            print("Selection incomplete.")
            return

        self.on_set_callback(selected_model, selected_datasets)


class FormattingView(ctk.CTkFrame):
    def __init__(self, master, vars, on_format_callback, **kwargs):
        super().__init__(master, **kwargs)
        self.vars = vars
        self.on_format_callback = on_format_callback

        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self, text="Dataset Formatting Manager", font=("", 20, "bold")).grid(
            row=0, column=0, columnspan=2, pady=10, padx=20, sticky="w")

        # 1. Model Selection (Base Model needed for formatting/tokenization)
        self.model_frame = ctk.CTkScrollableFrame(self, label_text="Select Tokenizer Model")
        self.model_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.model_var = ctk.StringVar()

        # 2. Raw Dataset Selection (Select multiple to format)
        self.dataset_frame = ctk.CTkScrollableFrame(self, label_text="Select Raw Datasets")
        self.dataset_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.dataset_vars = {}

        # 3. Action Button
        self.format_btn = ctk.CTkButton(
            self, text="✨ Run Formatting Process",
            command=self.submit_format,
            fg_color="#d88c00",  # Orange to distinguish from training
            hover_color="#b07200"
        )
        self.format_btn.grid(row=2, column=0, columnspan=2, pady=20, padx=20, sticky="ew")

        self.refresh_selectors()

    def refresh_selectors(self):
        for child in self.model_frame.winfo_children(): child.destroy()
        for child in self.dataset_frame.winfo_children(): child.destroy()

        # Populate Models (Tokenizer source)
        model_paths = [self.vars.REGULAR_MODEL_DIR, self.vars.VISION_MODEL_DIR,self.vars.LocalModel_DIR]
        for p in model_paths:
            if p.exists():
                for owner in p.iterdir():
                    if owner.is_dir():
                        for model in owner.iterdir():
                            if model.is_dir():
                                name = f"{owner.name}/{model.name}"
                                ctk.CTkRadioButton(self.model_frame, text=name, variable=self.model_var,
                                                   value=name).pack(anchor="w", pady=2, padx=10)

        # Populate Raw Datasets (Datasets folder, not formatted folder)
        if self.vars.DATASETS_DIR.exists():
            for d in self.vars.DATASETS_DIR.iterdir():
                if d.is_dir():
                    var = ctk.BooleanVar(value=False)
                    self.dataset_vars[d.name] = var
                    ctk.CTkCheckBox(self.dataset_frame, text=d.name, variable=var).pack(anchor="w", pady=2, padx=10)

    def submit_format(self):
        selected_model = self.model_var.get()
        selected_datasets = [name for name, var in self.dataset_vars.items() if var.get()]

        if not selected_model or not selected_datasets:
            print("Please select a model and at least one dataset.")
            return

        self.on_format_callback(selected_model, selected_datasets, self)


class ConfigView(ctk.CTkFrame):
    def __init__(self, master, vars, **kwargs):
        super().__init__(master, **kwargs)
        self.vars = vars
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # Editor takes most space

        # --- Control Header ---
        self.header = ctk.CTkFrame(self, fg_color="transparent")
        self.header.grid(row=0, column=0, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(self.header, text="Config Editor", font=("", 18, "bold")).pack(side="left")

        # Dropdown for file selection
        self.file_list = self.get_config_files()
        self.file_combo = ctk.CTkComboBox(self.header, values=self.file_list,
                                          command=self.load_selected_file, width=350)
        self.file_combo.pack(side="left", padx=20)

        # Action Buttons
        self.btn_save = ctk.CTkButton(self.header, text="💾 Save", width=80,
                                      fg_color="#2c6e49", command=self.save_current_file)
        self.btn_save.pack(side="right", padx=5)

        self.btn_reload = ctk.CTkButton(self.header, text="🔄 Reload", width=80,
                                        command=lambda: self.load_selected_file(self.file_combo.get()))
        self.btn_reload.pack(side="right", padx=5)

        # --- Code Editor ---
        self.editor = ctk.CTkTextbox(self, font=("Consolas", 12), undo=True, wrap="none")
        self.editor.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

        # Auto-load the first file if available
        if self.file_list:
            self.file_combo.set(self.file_list[0])
            self.load_selected_file(self.file_list[0])

    def get_config_files(self):
        """Scans the workspace for editable files."""
        files = []
        # Paths derived from your project structure
        targets = [
            self.vars.WORKSPACE / "configs",
            self.vars.WORKSPACE / "chat_template"
        ]

        for folder in targets:
            if folder.exists():
                for ext in ["*.json", "*.jinja"]:
                    files.extend([str(p) for p in folder.glob(ext)])
        return sorted(files)

    def load_selected_file(self, filepath):
        if not filepath or not os.path.exists(filepath):
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            self.editor.delete("1.0", "end")
            self.editor.insert("1.0", content)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file: {e}")

    def save_current_file(self):
        filepath = self.file_combo.get()
        if not filepath:
            return

        try:
            content = self.editor.get("1.0", "end-1c")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            messagebox.showinfo("Success", f"Saved: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

class CreateModelView(ctk.CTkFrame):
    def __init__(self, master, vars, on_create_callback, **kwargs):
        super().__init__(master, **kwargs)
        self.vars = vars
        self.on_create_callback = on_create_callback

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self, text="Model Capability Wrapper", font=("", 20, "bold")).grid(
            row=0, column=0, pady=10, padx=20, sticky="w")

        # 1. Base Model Selection
        self.model_frame = ctk.CTkScrollableFrame(self, label_text="Select Base Text Model")
        self.model_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.model_var = ctk.StringVar()

        # 2. Options Frame
        self.options_frame = ctk.CTkFrame(self)
        self.options_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(self.options_frame, text="Select Capability to Add:", font=("", 14, "bold")).pack(side="left",
                                                                                                       padx=10)

        self.mode_var = ctk.StringVar(value="conversation")
        ctk.CTkRadioButton(self.options_frame, text="Conversation (Qwen Wrapper)",
                           variable=self.mode_var, value="conversation").pack(side="left", padx=20)
        ctk.CTkRadioButton(self.options_frame, text="Vision (Vision Wrapper)",
                           variable=self.mode_var, value="vision").pack(side="left", padx=20)

        # 3. Action Button
        self.create_btn = ctk.CTkButton(
            self, text="🛠 Create Wrapped Model",
            command=self.submit_creation,
            fg_color="#5e35b1",  # Purple to distinguish
            hover_color="#4527a0"
        )
        self.create_btn.grid(row=3, column=0, pady=20, padx=20, sticky="ew")

        self.refresh_selectors()

    def refresh_selectors(self):
        for child in self.model_frame.winfo_children(): child.destroy()

        # Looking into LocalModel_DIR for base models (e.g., Qwen/Qwen1.5-0.5B-Chat)
        base_path = self.vars.LocalModel_DIR
        if base_path.exists():
            for owner in base_path.iterdir():
                if owner.is_dir():
                    for model in owner.iterdir():
                        if model.is_dir():
                            name = f"{owner.name}/{model.name}"
                            path = str(model)
                            ctk.CTkRadioButton(self.model_frame, text=name,
                                               variable=self.model_var, value=path).pack(anchor="w", pady=2, padx=10)

    def submit_creation(self):
        selected_path = self.model_var.get()
        mode = self.mode_var.get()

        if not selected_path:
            return

        self.on_create_callback(selected_path, mode, self)

class TrainView(ctk.CTkFrame):
    def __init__(self, master, vars, **kwargs):
        super().__init__(master, **kwargs)
        self.vars = vars

        self.grid_columnconfigure(0, weight=1)
        # The top terminal row takes all the extra space
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        # 1. TOP: The Internal Terminal for this page
        self.internal_log = ctk.CTkTextbox(self, font=("Consolas", 11), fg_color="#000000")
        self.internal_log.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")
        self.internal_log.insert("0.0", "--- Training Logs Will Appear Here ---\n")
        self.internal_log.configure(state="disabled")

        # 2. BOTTOM: The Status and Controls (Inverted)
        self.controls = ctk.CTkFrame(self, fg_color="transparent")
        self.controls.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")

        self.status_label = ctk.CTkLabel(
            self.controls,
            text="Ready to train. Ensure ApiCardSet.json is configured.",
            font=("", 13)
        )
        self.status_label.pack(pady=5)

        self.train_btn = ctk.CTkButton(
            self.controls,
            text="Start Fine-tuning",
            height=40,
            command=self.start_training,
            fg_color="#1b4332",
            hover_color="#081c15"
        )
        self.train_btn.pack(fill="x", padx=10)

    def write_internal(self, text):
        """Helper to write specifically to the top terminal of this page."""
        self.internal_log.configure(state="normal")
        self.internal_log.insert("end", text)
        self.internal_log.see("end")
        self.internal_log.configure(state="disabled")

    def start_training(self):
        from modules.train import FinetuneModel  # Local import to avoid circular issues

        self.train_btn.configure(state="disabled", text="Training...")
        self.status_label.configure(text="STATUS: Fine-tuning in progress...")
        self.write_internal("\nSystem: Starting Internal Finetune Engine...\n")

        def run():
            try:
                finetuner = FinetuneModel()

                # Custom stream to capture prints and send to the TOP terminal
                class LocalStream(io.TextIOBase):
                    def __init__(self, write_func): self.write_func = write_func

                    def write(self, s):
                        if s.strip(): self.write_func(s)
                        return len(s)

                with redirect_stdout(LocalStream(self.write_internal)):
                    finetuner.finetune_model()

                self.after(0, lambda: self.status_label.configure(text="STATUS: Finished successfully."))
            except Exception as e:
                self.after(0, lambda err=e: self.write_internal(f"\nError: {str(err)}\n"))
            finally:
                self.after(0, lambda: self.train_btn.configure(state="normal", text="Start Fine-tuning"))

        threading.Thread(target=run, daemon=True).start()