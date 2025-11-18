"""
Model utilities untuk loading dan generation
"""
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from typing import Dict, Optional
import os


class ModelManager:
    """Manage GPT-2 model loading, training, dan generation"""

    def __init__(
        self,
        model_name: str = "gpt2",
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None

        print(f"🖥️  Using device: {self.device}")

    def load_model(self, from_pretrained: bool = True) -> None:
        """
        Load model dan tokenizer

        Args:
            from_pretrained: Jika True, load dari pretrained (gpt2 atau fine-tuned)
        """
        if from_pretrained and self.model_path and os.path.exists(self.model_path):
            print(f"📦 Loading fine-tuned model from {self.model_path}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
        else:
            print(f"📦 Loading base model: {self.model_name}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

        # Move model to device
        self.model.to(self.device)

        print(f"✓ Model loaded successfully")

    def save_model(self, save_path: str) -> None:
        """Save model dan tokenizer"""
        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        print(f"✓ Model saved to {save_path}")

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        no_repeat_ngram_size: int = 3
    ) -> str:
        """
        Generate response dari prompt

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            no_repeat_ngram_size: Prevent repeating n-grams

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model belum di-load. Jalankan load_model() terlebih dahulu.")

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (remove prompt)
        generated_text = generated_text[len(prompt):].strip()

        return generated_text

    def train_model(
        self,
        train_dataset,
        output_dir: str = "./models/fine_tuned_gpt2",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 2,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        save_steps: int = 500,
        logging_steps: int = 50,
        save_total_limit: int = 2
    ) -> None:
        """
        Fine-tune model dengan training dataset

        Args:
            train_dataset: HuggingFace Dataset untuk training
            output_dir: Directory untuk save model
            Other args: Training hyperparameters
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model belum di-load. Jalankan load_model() terlebih dahulu.")

        print("🚀 Starting training...")

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # GPT-2 uses causal LM, not masked LM
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            logging_steps=logging_steps,
            logging_dir=f"{output_dir}/logs",
            report_to="none",  # Disable wandb, etc.
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        # Save final model
        self.save_model(output_dir)

        print("✅ Training complete!")


class RAGGenerator:
    """
    RAG Generator: Combines retrieval dengan generation
    """

    def __init__(self, model_manager: ModelManager, retriever):
        self.model_manager = model_manager
        self.retriever = retriever

    def generate_with_context(
        self,
        query: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        include_context_in_prompt: bool = True
    ) -> Dict:
        """
        Generate response dengan RAG (Retrieval + Generation)

        Args:
            query: User query
            max_new_tokens: Max tokens to generate
            temperature: Generation temperature
            include_context_in_prompt: Include retrieved context in prompt

        Returns:
            {
                'query': str,
                'answer': str,
                'context': str,
                'confidence': float,
                'method': str
            }
        """
        # Retrieve context
        retrieval_result = self.retriever.retrieve_context(query)

        # Build prompt
        if include_context_in_prompt and retrieval_result['context']:
            prompt = f"""Context: {retrieval_result['context']}

Question: {query}
Answer:"""
        else:
            prompt = f"Question: {query}\nAnswer:"

        # Generate response
        generated_answer = self.model_manager.generate_response(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        return {
            'query': query,
            'answer': generated_answer,
            'context': retrieval_result['context'],
            'confidence': retrieval_result['confidence'],
            'method': retrieval_result['method']
        }


# Usage example
if __name__ == "__main__":
    # Initialize model manager
    mm = ModelManager(model_name="gpt2")
    mm.load_model()

    # Test generation
    test_prompt = "Question: Bagaimana cara install Laravel?\nAnswer:"
    response = mm.generate_response(test_prompt, max_new_tokens=100)
    print(f"Generated: {response}")
