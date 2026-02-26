import customtkinter as ctk
from pathlib import Path
import sys
import threading


current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from modules.DataDownload import DataLoader
from modules.ApiDump import ApiCardSetup
from modules.DataModelPrepare import  Manager as DataManager
from modules.ModelUtils import CreateModel
from modules.variable import Variable
from modules.train import FinetuneModel
import io
from contextlib import redirect_stdout
from ui_components import ListFrame,DownloadHubView ,TaskSetterView,FormattingView,ConfigView,CreateModelView,TrainView


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Model DEV Studio")
        self.geometry("1100x600")
        self.vars = Variable()

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        # global terminal
        self.terminal = ctk.CTkTextbox(self, height=150, font=("Consolas", 11), fg_color="#1a1a1a")
        self.terminal.grid(row=1, column=1, sticky="ew", padx=10, pady=(0, 10))
        self.write_log("System Ready. Global Terminal Initialized...\n")

        # Views Dictionary
        self.views = {}
        self.setup_views()


        nav_items = [ k for k in self.views.keys() if k != "config" ]

        # Navigation
        for name in nav_items:
            ctk.CTkButton(self.sidebar, text=name.replace("_"," ").title(),
                          command=lambda n=name: self.show_view(n)).pack(pady=5, padx=10)
        spacer = ctk.CTkLabel(self.sidebar, text="")
        spacer.pack(expand=True, fill="both")
        self.config_btn = ctk.CTkButton(self.sidebar, text="Configuration",
                                        fg_color="#3d3d3d",
                                        command=lambda: self.show_view("config"))
        self.config_btn.pack(pady=20, padx=10, side="bottom")

        #show first pages
        self.show_view("download")

    def write_log(self, text):
        self.terminal.configure(state="normal")
        self.terminal.insert("end", text)
        self.terminal.see("end")
        self.terminal.configure(state="disabled")

    def setup_views(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)

        # 1, Download Model / Dataset
        self.views["download"] = DownloadHubView(container, self.handle_hf_download)

        # 2. Dataset View (Using Generic List)
        self.views["formatted_dataset"] = ListFrame(container, "Formatted Dataset",
                                           lambda: [d.name for d in self.vars.DATASET_FORMATTED_DIR.iterdir() if
                                                    d.is_dir()] if self.vars.DATASET_FORMATTED_DIR.exists() else [])
        # 3. Model View
        self.views["custom_model"] = ListFrame(container,"Custom Model",
                                               lambda: [
                                                   sd.name
                                                   for path in [self.vars.REGULAR_MODEL_DIR, self.vars.VISION_MODEL_DIR]
                                                   if path.exists()
                                                   for sd in path.iterdir()
                                                   if sd.is_dir()
                                               ])

        # 4. set task for train job
        self.views["Set_Task"] = TaskSetterView(container, self.vars, self.handle_set_apicard)

        # 5. Training View (Using Generic Logger)
        self.views["train"] = TrainView(container, self.handle_internal_train)

        # 6. Create Model
        self.views["create_model"] = CreateModelView(container, self.vars, self.handle_model_creation)
        # 7. Formatting View (Using Generic Logger)
        self.views["Dataset Format"] = FormattingView(container, self.vars, self.handle_data_formatting)


        # 7. Config
        self.views["config"] = ConfigView(container, self.vars)

        for v in self.views.values(): v.grid(row=0, column=0, sticky="nsew")

    def show_view(self, name):
        for v in self.views.values(): v.grid_remove()
        self.views[name].grid()

    # Logic Handlers (Passed to the UI components)
    def handle_training(self, frame):
        frame.btn.configure(state="disabled")
        # run_task_in_background("python modules/train.py", frame.write_log, lambda: frame.btn.configure(state="normal"))

    def handle_formatting(self, frame):
        frame.btn.configure(state="disabled")
        # run_task_in_background("python modules/DataModelPrepare.py", frame.write_log,
        #                        lambda: frame.btn.configure(state="normal"))

    def handle_hf_download(self, view):
        repo_id = view.repo_entry.get().strip()
        is_model = view.type_var.get() == "Model"

        if not repo_id:
            self.write_log("System: Please enter a Repository ID.\n")
            return

        view.dl_btn.configure(state="disabled")
        self.write_log(f"Request: Downloading {repo_id}...\n")

        # Run in thread so UI stays responsive
        threading.Thread(
            target=self.run_download_task,
            args=(
                repo_id,
                is_model,
                self.write_log,  # Send logs to global terminal
                lambda success: view.dl_btn.configure(state="normal")  # Re-enable button on finish
            ),
            daemon=True
        ).start()

    def run_download_task(self,repo_id, is_model, log_callback, finish_callback):
        """Handles the HF Download logic in a thread."""
        vars = Variable()
        api = vars.hf_api
        setcard = ApiCardSetup()
        downloader = DataLoader()

        try:
            log_callback(f"Searching for {repo_id}...\n")
            if is_model:
                item = api.list_models(model_name=repo_id, limit=1)
                if not item:
                    log_callback(f"Error: Model {repo_id} not found.\n")
                    return finish_callback(False)
                list_download = setcard.set(item, None)
            else:
                item = api.list_datasets(dataset_name=repo_id, limit=1)
                if not item:
                    log_callback(f"Error: Dataset {repo_id} not found.\n")
                    return finish_callback(False)
                list_download = setcard.set(None, item)

            log_callback(f"Starting download: {repo_id}\n")
            downloader.run(list_download)
            log_callback("Download Complete!\n")
            finish_callback(True)
        except Exception as e:
            log_callback(f"Download Failed: {str(e)}\n")
            finish_callback(False)

    def handle_set_apicard(self, model_name, dataset_names):
        self.write_log(f"System: Updating ApiCard for Model [{model_name}] with {len(dataset_names)} datasets...\n")

        try:
            # Replicating your old logic to build the dictionary
            list_download_dict = {'model': {}}
            list_download_dict['model'][model_name] = {ds_name: "" for ds_name in dataset_names}

            set_card = ApiCardSetup()
            # This updates the JSON file without starting the training process
            set_card.set(from_repository=list_download_dict)

            self.write_log(f"Success: ApiCardSet.json updated. You can now go to 'Training' to start.\n")
        except Exception as e:
            self.write_log(f"Error: Failed to update ApiCard: {str(e)}\n")

    def show_view(self, name):
        for v in self.views.values(): v.grid_remove()
        self.views[name].grid()
        # Refresh the lists whenever we switch to this view
        if name == "set_task":
            self.views[name].refresh_selectors()
        elif name == "format":
            self.views[name].refresh_selectors()
        elif name == "create_model":
            # Updated from refresh_selector() to refresh_selectors()
            self.views[name].refresh_selectors()
        elif name == "config":
            new_files = self.views[name].get_config_files()
            self.views[name].file_combo.configure(values=new_files)

            self.write_log("System: Config editor synchronized.\n")

    def handle_data_formatting(self, model_name, dataset_names, view):
        self.write_log(f"System: Starting Formatting with Model [{model_name}]...\n")
        view.format_btn.configure(state="disabled", text="Formatting in Progress...")

        def run():
            try:
                # 1. Prepare data structure for the Manager
                # This mimics the format required by your DataModelPrepare module
                list_data = {'model': {model_name: {ds: "" for ds in dataset_names}}}

                # 2. Call the module
                data_manager = DataManager()
                # Assuming dataset_prepare is the main entry point in your module
                data_manager.dataset_prepare(list_data)

                self.after(0, lambda: self.write_log("Success: Data formatting completed!\n"))
            except Exception as e:
                self.after(0, lambda err=e : self.write_log(f"Error during formatting: {str(err)}\n"))
            finally:
                self.after(0, lambda: view.format_btn.configure(state="normal", text="Run Formatting Process"))

        threading.Thread(target=run, daemon=True).start()

    def handle_model_creation(self, model_path, mode, view):
        self.write_log(f"System: Initiating {mode} model creation from {model_path}...\n")
        view.create_btn.configure(state="disabled", text="Processing...")

        def run():
            try:
                # Map the UI mode to your class types
                model_type = "conversation-model" if mode == "conversation" else "vision-model"

                creator = CreateModel(Path(model_path), model_type)

                if mode == "conversation":
                    self.write_log("System: Adding Conversation wrapper...\n")
                    creator.add_conversation()
                    creator.save_regular_model()
                else:
                    self.write_log("System: Adding Vision capability wrapper...\n")
                    creator.add_vision()
                    creator.save_vision_model()

                self.after(0,
                           lambda: self.write_log(f"Success: {mode.title()} model created and saved to repository!\n"))
            except Exception as e:
                self.after(0, lambda err=e: self.write_log(f"Error: Model creation failed: {str(err)}\n"))
            finally:
                self.after(0, lambda:  view.create_btn.configure(state="normal", text="Create Wrapped Model"))

        threading.Thread(target=run, daemon=True).start()

    def handle_internal_train(self, view):
        """Executes the FinetuneModel logic in a background thread."""
        self.write_log("System: Initializing Fine-tuning engine...\n")
        view.train_btn.configure(state="disabled", text="Training in Progress...")

        def run_training():
            # Create a stream to capture internal print statements
            try:
                # 1. Initialize the module
                finetuner = FinetuneModel()

                # 2. Redirect internal prints to the global terminal
                # We create a small helper to pipe the stream line-by-line
                class TerminalStream(io.TextIOBase):
                    def __init__(self, log_func): self.log_func = log_func

                    def write(self, s):
                        if s.strip(): self.log_func(s)
                        return len(s)

                with redirect_stdout(TerminalStream(self.write_log)):
                    self.write_log("System: Starting finetune_model() execution...\n")
                    # 3. Call your old implementation directly
                    finetuner.finetune_model()

                self.after(0, lambda: self.write_log("\nSuccess: Training process completed!\n"))
            except Exception as e:
                self.after(0, lambda err=e: self.write_log(f"\nError: Training failed: {str(err)}\n"))
            finally:
                self.after(0, lambda: view.train_btn.configure(state="normal", text="Start Fine-tuning"))

        threading.Thread(target=run_training, daemon=True).start()



if __name__ == "__main__":
    App().mainloop()