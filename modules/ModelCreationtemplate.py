class CustomConfig(PretrainedConfig):
    
    
    def __init__(self,model_types,architectures, **kwargs):
        model_types = model_types
        architectures = [architectures]
        super().__init__(model_types=model_types, architectures=architectures, **kwargs)

class ModelTemplate:
    def __init__(self,register_name):
        self.register_name = register_name
        self.model = NewModel()
    @property
    def is_gradient_checkpointing(self) -> bool:
        return self._is_gradient_checkpointing

    @is_gradient_checkpointing.setter
    def is_gradient_checkpointing(self, value: bool):
        self._is_gradient_checkpointing = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        self.is_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        self.is_gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
            
    def save_pretrained(self, save_directory, **kwargs):
        """Save the model."""
        # Save PEFT configuration if it exists
        if hasattr(self.model, 'peft_config'):
            self.model.save_pretrained(save_directory, **kwargs)
        else:
            super().save_pretrained(save_directory, **kwargs)


class NewModel:
    def __init__(self):
        pass