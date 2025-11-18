"""
Auto fine-tuning system dengan approval workflow
"""
import os
import json
from typing import List, Dict
from datetime import datetime
from data_processing import DataProcessor
from model_utils import ModelManager


class AutoFineTuner:
    """
    Automatic fine-tuning system dengan human-in-the-loop approval
    """

    def __init__(
        self,
        model_manager: ModelManager,
        base_dataset_path: str = "./data/raw/laravel_qa_dataset.json",
        approved_qa_path: str = "./data/raw/approved_qa.json",
        training_output_dir: str = "./models/auto_finetuned"
    ):
        self.model_manager = model_manager
        self.base_dataset_path = base_dataset_path
        self.approved_qa_path = approved_qa_path
        self.training_output_dir = training_output_dir

        self.training_history = []

    def prepare_training_data(self, approved_qa: List[Dict]) -> Dict:
        """
        Prepare training data dari approved QA pairs

        Returns:
            Dict dengan info tentang prepared data
        """
        # Load base dataset
        base_data = []
        if os.path.exists(self.base_dataset_path):
            with open(self.base_dataset_path, 'r', encoding='utf-8') as f:
                base_data = json.load(f)

        print(f"📊 Base dataset: {len(base_data)} QA pairs")
        print(f"📊 Approved new QA: {len(approved_qa)} pairs")

        # Merge datasets
        merged_data = base_data.copy()

        # Add approved QA (avoid duplicates)
        existing_questions = {qa['question'].lower() for qa in base_data}

        new_count = 0
        for qa in approved_qa:
            if qa['question'].lower() not in existing_questions:
                merged_data.append(qa)
                new_count += 1

        print(f"✅ Added {new_count} new unique QA pairs")
        print(f"📊 Total training data: {len(merged_data)} pairs")

        # Save merged dataset
        merged_path = self.base_dataset_path.replace('.json', '_merged.json')
        with open(merged_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)

        return {
            'base_count': len(base_data),
            'approved_count': len(approved_qa),
            'new_unique': new_count,
            'total': len(merged_data),
            'merged_path': merged_path
        }

    def start_finetuning(
        self,
        approved_qa: List[Dict],
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5
    ) -> Dict:
        """
        Start automatic fine-tuning dengan approved data

        Args:
            approved_qa: List of approved QA pairs
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate

        Returns:
            Training results
        """
        print("\n" + "=" * 60)
        print("🚀 Starting Automatic Fine-Tuning")
        print("=" * 60)

        # 1. Prepare data
        data_info = self.prepare_training_data(approved_qa)

        # 2. Process data for training
        processor = DataProcessor(
            raw_data_path=data_info['merged_path'],
            processed_data_path=self.approved_qa_path.replace('.json', '_processed.json')
        )

        processor.process_and_save()
        print("\n✅ Data processed for training")

        # 3. Create dataset
        print("\n📦 Creating training dataset...")
        train_dataset = processor.create_dataset_for_training(
            tokenizer=self.model_manager.tokenizer,
            max_length=256
        )

        print(f"✅ Training dataset: {len(train_dataset)} samples")

        # 4. Train model
        print("\n🏋️  Training model...")
        training_start = datetime.now()

        self.model_manager.train_model(
            train_dataset=train_dataset,
            output_dir=self.training_output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=100,
            logging_steps=50
        )

        training_duration = (datetime.now() - training_start).total_seconds()

        print(f"\n✅ Training completed in {training_duration:.2f} seconds")

        # 5. Save training history
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'data_info': data_info,
            'training_params': {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            },
            'duration_seconds': training_duration,
            'model_output_dir': self.training_output_dir
        }

        self.training_history.append(training_record)

        # Save history
        history_path = os.path.join(self.training_output_dir, 'training_history.json')
        os.makedirs(os.path.dirname(history_path), exist_ok=True)

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Training history saved to {history_path}")

        print("\n" + "=" * 60)
        print("🎉 Auto Fine-Tuning Complete!")
        print("=" * 60)
        print(f"\n📊 Summary:")
        print(f"   New QA pairs added: {data_info['new_unique']}")
        print(f"   Total training data: {data_info['total']}")
        print(f"   Training duration: {training_duration:.2f}s")
        print(f"   Model saved to: {self.training_output_dir}")

        return training_record

    def schedule_periodic_finetuning(
        self,
        check_interval_hours: int = 24,
        min_new_qa_threshold: int = 10
    ):
        """
        Schedule periodic fine-tuning (untuk production)

        Args:
            check_interval_hours: Check for new approved QA every N hours
            min_new_qa_threshold: Only fine-tune if at least N new QA pairs approved
        """
        print(f"📅 Scheduled periodic fine-tuning:")
        print(f"   Check interval: {check_interval_hours} hours")
        print(f"   Minimum new QA: {min_new_qa_threshold} pairs")

        # This would be implemented with a scheduler like APScheduler
        # For now, just return the config
        return {
            'check_interval_hours': check_interval_hours,
            'min_new_qa_threshold': min_new_qa_threshold,
            'note': 'Use APScheduler or Celery for production implementation'
        }

    def get_training_stats(self) -> Dict:
        """Get statistics about training history"""
        if not self.training_history:
            return {
                'total_trainings': 0,
                'message': 'No training history yet'
            }

        total_qa_added = sum(t['data_info']['new_unique'] for t in self.training_history)
        total_duration = sum(t['duration_seconds'] for t in self.training_history)

        return {
            'total_trainings': len(self.training_history),
            'total_qa_added': total_qa_added,
            'total_training_time': total_duration,
            'avg_training_time': total_duration / len(self.training_history),
            'last_training': self.training_history[-1]['timestamp'],
            'latest_model': self.training_history[-1]['model_output_dir']
        }


# Example usage
if __name__ == "__main__":
    from model_utils import ModelManager

    # Initialize
    model_manager = ModelManager(model_name="gpt2")
    model_manager.load_model()

    auto_tuner = AutoFineTuner(model_manager)

    # Example approved QA
    approved_qa = [
        {
            'id': 1,
            'question': 'Bagaimana cara deploy Laravel ke production?',
            'answer': '1. Setup server dengan PHP 8.1+, 2. Install dependencies, 3. Configure .env...',
            'category': 'deployment',
            'difficulty': 'advanced'
        }
    ]

    # Start fine-tuning
    # result = auto_tuner.start_finetuning(approved_qa, num_epochs=2)

    print("\n📊 Training Stats:")
    print(json.dumps(auto_tuner.get_training_stats(), indent=2))
