"""
Configuration loader untuk Laravel RAG LLM
"""
import json
import os
from pathlib import Path


class ConfigLoader:
    """Load dan manage configuration dari config file"""

    def __init__(self, config_path: str = "./configs/config.json"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration dari JSON file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file tidak ditemukan: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def get(self, key: str, default=None):
        """Get config value dengan dot notation (e.g., 'model.name')"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_model_config(self) -> dict:
        """Get model configuration"""
        return self.config.get('model', {})

    def get_training_config(self) -> dict:
        """Get training configuration"""
        return self.config.get('training', {})

    def get_data_config(self) -> dict:
        """Get data configuration"""
        return self.config.get('data', {})

    def get_retrieval_config(self) -> dict:
        """Get retrieval configuration"""
        return self.config.get('retrieval', {})

    def get_generation_config(self) -> dict:
        """Get generation configuration"""
        return self.config.get('generation', {})


# Usage example
if __name__ == "__main__":
    config = ConfigLoader()
    print(f"Model name: {config.get('model.name')}")
    print(f"Training epochs: {config.get('training.num_train_epochs')}")
