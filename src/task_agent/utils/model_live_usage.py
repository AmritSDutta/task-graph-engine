import logging
import threading
from collections import defaultdict
from typing import Dict, List, Union

from task_agent.config import settings


class ModelLiveUsage:
    def __init__(self):
        self.model_usage: Dict[str, int] = defaultdict(lambda: 0)
        self.model_token_usage: Dict[str, float] = defaultdict(lambda: settings.TOKEN_USAGE_LOG_BASE)

    def add_model_usage(self, model_name: str, usage: int = 1) -> None:
        self.model_usage[model_name] = self.model_usage[model_name] + usage

    def get_models_usage(self, model_names: List[str]) -> Dict[str, int]:
        """Returns a dictionary with the current usage values for the specified models."""
        return {name: self.model_usage[name] for name in model_names}

    def get_model_usage(self, model_name: str) -> int:
        """Returns a dictionary with the current usage values for the specified models."""
        return self.model_usage[model_name]

    def log_model_usage(self) -> None:
        logging.info(f"Models live usage: \n Calls:\n {self.model_usage}\n Tokens: \n {self.model_token_usage}")

    def add_model_token_usage(self, model_name: str, usage: Union[int, float] = 1e-9) -> None:
        self.model_token_usage[model_name] += float(usage)

    def get_models_token_usage(self, model_names: List[str]) -> Dict[str, float]:
        """Returns a dictionary with the current usage values for the specified models."""
        return {name: self.model_token_usage[name] for name in model_names}

    def get_model_token_usage(self, model_name: str) -> float:
        """Returns a dictionary with the current usage values for the specified models."""
        return self.model_token_usage[model_name]


class ModelLiveUsageSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super(ModelLiveUsageSingleton, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Ensure initialization happens only once
        if getattr(self, '_initialized', False):
            return

        with self._lock:
            if getattr(self, '_initialized', False):
                return

            self._model_usage_instance = ModelLiveUsage()
            self._initialized = True

    def get_instance(self) -> ModelLiveUsage:
        """Returns the ModelLiveUsage instance."""
        return self._model_usage_instance


# Convenience function to get the singleton instance
def get_model_usage_singleton() -> ModelLiveUsage:
    """Returns the singleton instance of ModelLiveUsage."""
    singleton = ModelLiveUsageSingleton()
    return singleton.get_instance()
