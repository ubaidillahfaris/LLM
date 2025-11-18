"""
Data processing utilities untuk Laravel RAG LLM
"""
import json
import os
from typing import List, Dict, Tuple
from datasets import Dataset


class DataProcessor:
    """Process data untuk training dan inference"""

    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    def load_raw_data(self) -> List[Dict]:
        """Load raw QA dataset"""
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data tidak ditemukan: {self.raw_data_path}")

        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_processed_data(self) -> List[Dict]:
        """Load processed training data"""
        if not os.path.exists(self.processed_data_path):
            raise FileNotFoundError(f"Processed data tidak ditemukan: {self.processed_data_path}")

        with open(self.processed_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def convert_qa_to_training_format(self, qa_data: List[Dict]) -> List[Dict]:
        """
        Convert QA dataset ke training format
        Format: {"prompt": "Q: ... A:", "completion": " ..."}
        """
        training_data = []

        for item in qa_data:
            training_item = {
                "prompt": f"Q: {item['question']}\nA:",
                "completion": f" {item['answer']}"
            }
            training_data.append(training_item)

        return training_data

    def save_processed_data(self, data: List[Dict]) -> None:
        """Save processed data to JSON file"""
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)

        with open(self.processed_data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved {len(data)} training samples to {self.processed_data_path}")

    def create_dataset_for_training(self, tokenizer, max_length: int = 256) -> Dataset:
        """
        Create HuggingFace Dataset untuk training
        """
        # Load processed data
        data = self.load_processed_data()

        # Combine prompt and completion
        texts = [item['prompt'] + item['completion'] for item in data]

        # Create dataset
        dataset = Dataset.from_dict({"text": texts})

        # Tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        return tokenized_dataset

    def process_and_save(self) -> None:
        """
        Main processing pipeline:
        1. Load raw QA data
        2. Convert to training format
        3. Save processed data
        """
        print("📊 Processing data...")

        # Load raw data
        raw_data = self.load_raw_data()
        print(f"✓ Loaded {len(raw_data)} QA pairs")

        # Convert to training format
        training_data = self.convert_qa_to_training_format(raw_data)
        print(f"✓ Converted to training format")

        # Save processed data
        self.save_processed_data(training_data)

        print("✅ Data processing complete!")


# Usage example
if __name__ == "__main__":
    processor = DataProcessor(
        raw_data_path="./data/raw/laravel_qa_dataset.json",
        processed_data_path="./data/processed/training_data.json"
    )
    processor.process_and_save()
