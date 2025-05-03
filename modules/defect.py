#all model download and dataset download error log store as json file to be filter an unusable model and dataset for installed.json
from pathlib import Path
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

class Report:
    def __init__(self):
        self.report_file = 'problem.txt'
        self.report_path = Path(__file__).parent.parent.absolute() / 'report' / self.report_file
        self.report_path.touch(exist_ok=True)
        
    def store_problem(self,model=None,dataset=None):
        if model:
            with open(self.report_path,'a') as f:
                f.write(f"Model: {model}\n")
        if dataset:
            with open(self.report_path,'a') as f:
                f.write(f"Dataset: {dataset}\n")
        print(f"{Fore.RED}Problem stored in {self.report_path}{Style.RESET_ALL}")
    
    def load_report(self):
        if self.report_path.exists():
            with open(self.report_path,'r') as f:
                return f.read()
        else:
            return None
                
