import abc

class BaseModel():
    
    @abc.abstractmethod
    def fetch_vit(self):
        pass

    @abc.abstractmethod
    def fetch_llm(self):
        pass

    @abc.abstractmethod
    def fetch_proj(self):
        pass

    @abc.abstractmethod
    def vision_preprocess(self, image):
        pass
    
    @abc.abstractmethod
    def language_preprocess(self, text):
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def eval(self):
        if hasattr(self, "model") and hasattr(self.model, "eval"):
            self.model.eval()
        return self

    def train(self, mode=True):
        if hasattr(self, "model") and hasattr(self.model, "train"):
            self.model.train(mode)
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)