from abc import ABC, abstractmethod
from typing import List, Dict, Any


class InferenceEngine(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = self.load_model()

    @abstractmethod
    def load_model(self) -> Any:
        pass

    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        pass

    @abstractmethod
    def inference(self, preprocessed_data: Any) -> Any:
        pass

    @abstractmethod
    def postprocess(self, inference_output: Any) -> Any:
        pass

 
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.config.get("model_name"),
            "model_version": self.config.get("model_version"),
            "model_type": self.config.get("model_type"),
            "model_framework": self.config.get("model_framework"),
            "model_framework_version": self.config.get("model_framework_version"),
        }
