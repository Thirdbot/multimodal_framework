from modules.models.ConversationModel import ConversationModel
from modules.ModelCreationtemplate import  ModelConfig,create_model,CustomModelConfig
from modules.ModelUtils import CreateModel
from transformers import AutoModelForCausalLM, AutoConfig
import jinja2

from modules.variable import Variable
from transformers import AutoTokenizer
vars = Variable()

chat_template_file = vars.chat_template_path / "chat_template_conversation.jinja"

def _read_template_str(template_path) -> str:
        template_file = template_path
        with open(template_file, "r", encoding="utf-8") as f:
            return f.read()
        
conversation_folder = vars.REGULAR_MODEL_DIR
repo_folder = vars.LocalModel_DIR

model_config = CustomModelConfig(
    base_config=ModelConfig(
    model_type="conversation-model",
    hidden_size=2048,
    architectures = [
    "Qwen"
  ],
    model_name="ConversationModel",
))


oldmodel_path = repo_folder /"Qwen"/ "Qwen1.5-0.5B-Chat"


# Load a tokenizer (e.g., from a pretrained model)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.chat_template = _read_template_str(chat_template_file)
tokenizer.pad_token = tokenizer.eos_token


# Create the conversation model with tokenizer
conver = ConversationModel(model_config, tokenizer=tokenizer)

# Save directly (tokenizer will be saved automatically)
conver.save_pretrained(conversation_folder / "newModel")
conver.save_pretrained(repo_folder / "newModel")

print("Successfully created and saved the conversation model. to "+ str(conversation_folder / "newModel"))



###attach text-based model to conversation model with conversationwrapper to forward

create_conver_model = CreateModel(oldmodel_path, "conversation-model")
create_conver_model.add_conversation()
create_conver_model.save_regular_model()


###attach text-based model to vision model with visionmodelwrapper to add Vision capability

create_vision_model = CreateModel(oldmodel_path, "vision-model")
create_vision_model.add_vision()
create_vision_model.save_vision_model()