from modules.variable import Variable

vars = Variable()

#create config folders
def create_config_folders():
    DMConfig_DIR = vars.DMConfig_DIR
    DMConfig_DIR.mkdir(parents=True, exist_ok=True)
    SavedConfigsfile= vars.SAVED_CONFIG_Path
    SavedConfigsfile.touch(exist_ok=True)