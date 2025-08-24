# ----------------------------------------------------------------------------
# Fine-Tuned Model System for Financial QA
# Features continual learning, domain adaptation, and comprehensive evaluation
# ----------------------------------------------------------------------------

# -------------------
# Importing libraries
# -------------------
import time
import json
import torch
import logging
import evaluate
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    pipeline, TextClassificationPipeline
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for fine-tuning"""
    model_name: str = "distilgpt2"
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 512
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100

class FinancialQADataset:
    """Dataset preparation for financial Q&A fine-tuning"""

    def __init__(self, qa_pairs: List[Dict[str, str]], max_length: int = 512):
        self.qa_pairs = qa_pairs
        self.max_length = max_length

    def prepare_for_fine_tuning(self) -> Dataset:
        """Convert Q&A pairs to fine-tuning format"""
        formatted_data = []

        for qa in self.qa_pairs:

            # -------------------------------
            # Format 1: Question-Answer pairs
            # -------------------------------
            formatted_data.append({
                'text': f"Question: {qa['question']}\nAnswer: {qa['answer']}",
                'question': qa['question'],
                'answer': qa['answer'],
                'category': qa.get('category', 'general'),
                'source': qa.get('source', 'unknown')
            })

            # --------------------------------------
            # Format 2: Instruction-following format
            # --------------------------------------
            formatted_data.append({
                'text': f"Instruction: Answer the following financial question.\nQuestion: {qa['question']}\nResponse: {qa['answer']}",
                'question': qa['question'],
                'answer': qa['answer'],
                'category': qa.get('category', 'general'),
                'source': qa.get('source', 'unknown')
            })

        return Dataset.from_list(formatted_data)

    def prepare_for_classification(self) -> Dataset:
        """Prepare data for question classification (continual learning)"""
        formatted_data = []

        for qa in self.qa_pairs:
            formatted_data.append({
                'text': qa['question'],
                'label': qa.get('category', 'general'),
                'answer': qa['answer']
            })

        return Dataset.from_list(formatted_data)

class ContinualLearningSystem:
    """Continual learning system for domain adaptation"""

    def __init__(self, base_model_name: str = "distilgpt2"):
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.learning_history = []

    def initialize_model(self):
        """Initialize the base model and tokenizer"""
        logger.info(f"Initializing model: {self.base_model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)

        # --------------------------------------------
        # Initialize classifier for continual learning
        # --------------------------------------------
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(set(['revenue', 'net_income', 'total_assets', 'company_info', 'business_segments']))
        )

        logger.info("Model initialization complete")

    def continual_fine_tune(self, new_data: Dataset, config: TrainingConfig):
        """Fine-tune model on new data while preserving prior knowledge"""
        logger.info("Starting continual fine-tuning...")

        # ------------
        # Prepare data
        # ------------
        tokenized_data = new_data.map(
            lambda x: self.tokenizer(
                x['text'],
                truncation=True,
                padding=True,
                max_length=config.max_length,
                return_tensors="pt"
            ),
            batched=True
        )

        # ------------------
        # Training arguments
        # ------------------
        training_args = TrainingArguments(
            output_dir=f"./continual_learning_{int(time.time())}",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            num_train_epochs=config.num_epochs,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            logging_steps=config.logging_steps,
            save_total_limit=2,
            # load_best_model_at_end=True,  # Removed for compatibility - requires matching eval strategy
            # evaluation_strategy="steps",  # Removed for compatibility with older transformers
            dataloader_pin_memory=False,
        )

        # -------------
        # Data collator
        # -------------
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # -------
        # Trainer
        # -------
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_data,
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        # Save model
        output_dir = f"./continual_model_{int(time.time())}"
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # -----------------------
        # Update learning history
        # -----------------------
        self.learning_history.append({
            'timestamp': time.time(),
            'data_size': len(new_data),
            'config': config.__dict__,
            'output_dir': output_dir
        })

        logger.info(f"Continual fine-tuning complete. Model saved to {output_dir}")

        return output_dir

    def adapt_to_new_domain(self, new_financial_data: List[Dict[str, str]]):
        """Adapt model to new financial domain data"""
        logger.info("Adapting model to new financial domain...")

        # -------------------
        # Prepare new dataset
        # -------------------
        new_dataset = FinancialQADataset(new_financial_data)
        formatted_data = new_dataset.prepare_for_fine_tuning()

        # ----------------------------------------------------------
        # Fine-tune with smaller learning rate for domain adaptation
        # ----------------------------------------------------------
        config = TrainingConfig(
            learning_rate=1e-5,  # Lower learning rate for adaptation
            num_epochs=2,        # Fewer epochs to avoid catastrophic forgetting
            batch_size=2,        # Smaller batch size
            max_length=512,
            warmup_steps=25,
            weight_decay=0.01,
            gradient_accumulation_steps=1,
            save_steps=50,
            eval_steps=50,
            logging_steps=25
        )

        return self.continual_fine_tune(formatted_data, config)

class FineTunedSystem:
    """Complete fine-tuned model system"""

    def __init__(self,
                 model_name: str = "distilgpt2",
                 model_path: Optional[str] = None):

        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.continual_learning = ContinualLearningSystem(model_name)

        # ----------
        # Initialize
        # ----------
        self.continual_learning.initialize_model()
        self.model = self.continual_learning.model
        self.tokenizer = self.continual_learning.tokenizer

        # Move to device (CUDA if is_available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info(f"Fine-tuned system initialized on {self.device}")

    def load_fine_tuned_model(self, model_path: str):
        """Load a previously fine-tuned model"""
        logger.info(f"Loading fine-tuned model from {model_path}")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.to(self.device)
            logger.info("Fine-tuned model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            logger.info("Using base model instead")

    def generate_answer(self, question: str, max_length: int = 200) -> str:
        """Generate answer using fine-tuned model"""

        # -------------
        # Create prompt
        # -------------
        prompt = f"Question: {question}\nAnswer:"

        # --------
        # Tokenize
        # --------
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(self.device)

        # --------
        # Generate
        # --------
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # ---------------
        # Decode response
        # ---------------
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # ----------------------------
        # Extract only the answer part
        # ----------------------------
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response[len(prompt):].strip()

        return answer

    def answer_question(self, question: str) -> Dict:
        """Main method to answer a question using fine-tuned model"""
        start_time = time.time()

        # ---------------
        # Generate answer
        # ---------------
        answer = self.generate_answer(question)

        # -------------------------------------------------------------------
        # Calculate confidence (heuristic based on answer length & coherence)
        # -------------------------------------------------------------------
        confidence = min(0.9, 0.3 + len(answer.split()) * 0.02)

        response_time = time.time() - start_time

        return {
            'answer': answer,
            'confidence': confidence,
            'method': 'fine_tuned',
            'response_time': response_time,
            'sources': ['fine_tuned_model']
        }

    def fine_tune_on_data(self, qa_pairs: List[Dict[str, str]], config: TrainingConfig):
        """Fine-tune the model on Q&A data"""
        logger.info("Starting fine-tuning process...")

        # ---------------
        # Prepare dataset
        # ---------------
        dataset = FinancialQADataset(qa_pairs)
        formatted_data = dataset.prepare_for_fine_tuning()

        # ---------
        # Fine-tune
        # ---------
        output_dir = self.continual_learning.continual_fine_tune(formatted_data, config)

        # -------------------------
        # Load the fine-tuned model
        # -------------------------
        self.load_fine_tuned_model(output_dir)

        return output_dir

class ModelEvaluator:
    """Comprehensive evaluation of fine-tuned models"""

    def __init__(self):
        self.metrics = {}
        self.rouge = evaluate.load('rouge')

    def evaluate_baseline(self, model, tokenizer, test_questions: List[str],
                         ground_truth: List[str]) -> Dict:
        """Evaluate baseline model performance"""
        logger.info("Evaluating baseline model...")

        predictions = []
        response_times = []

        for question, truth in zip(test_questions, ground_truth):
            start_time = time.time()

            # ---------------
            # Generate answer
            # ---------------
            prompt = f"Question: {question}\nAnswer:"
            inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()

            predictions.append(answer)
            response_times.append(time.time() - start_time)

        # -----------------
        # Calculate metrics
        # -----------------
        rouge_scores = self.rouge.compute(predictions=predictions, references=ground_truth)

        # -----------------------------
        # Simple accuracy (exact match)
        # -----------------------------
        exact_matches = sum(1 for pred, truth in zip(predictions, ground_truth)
                           if pred.lower().strip() == truth.lower().strip())
        accuracy = exact_matches / len(test_questions)

        self.metrics['baseline'] = {
            'rouge_scores': rouge_scores,
            'accuracy': accuracy,
            'avg_response_time': np.mean(response_times),
            'predictions': predictions,
            'response_times': response_times
        }

        return self.metrics['baseline']

    def evaluate_fine_tuned(self, fine_tuned_system: FineTunedSystem,
                           test_questions: List[str], ground_truth: List[str]) -> Dict:
        """Evaluate fine-tuned model performance"""
        logger.info("Evaluating fine-tuned model...")

        predictions = []
        response_times = []
        confidences = []

        for question, truth in zip(test_questions, ground_truth):
            response = fine_tuned_system.answer_question(question)
            predictions.append(response['answer'])
            response_times.append(response['response_time'])
            confidences.append(response['confidence'])

        # -----------------
        # Calculate metrics
        # -----------------
        rouge_scores = self.rouge.compute(predictions=predictions, references=ground_truth)

        # -----------------------------
        # Simple accuracy (exact match)
        # -----------------------------
        exact_matches = sum(1 for pred, truth in zip(predictions, ground_truth)
                           if pred.lower().strip() == truth.lower().strip())
        accuracy = exact_matches / len(test_questions)

        self.metrics['fine_tuned'] = {
            'rouge_scores': rouge_scores,
            'accuracy': accuracy,
            'avg_response_time': np.mean(response_times),
            'avg_confidence': np.mean(confidences),
            'predictions': predictions,
            'response_times': response_times,
            'confidences': confidences
        }

        return self.metrics['fine_tuned']

    def compare_models(self) -> Dict:
        """Compare baseline and fine-tuned models"""
        if 'baseline' not in self.metrics or 'fine_tuned' not in self.metrics:
            raise ValueError("Both baseline and fine-tuned evaluations must be completed first")

        comparison = {
            'accuracy_improvement': self.metrics['fine_tuned']['accuracy'] - self.metrics['baseline']['accuracy'],
            'speed_improvement': self.metrics['baseline']['avg_response_time'] - self.metrics['fine_tuned']['avg_response_time'],
            'rouge_improvement': {
                'rouge1': self.metrics['fine_tuned']['rouge_scores']['rouge1'] - self.metrics['baseline']['rouge_scores']['rouge1'],
                'rouge2': self.metrics['fine_tuned']['rouge_scores']['rouge2'] - self.metrics['baseline']['rouge_scores']['rouge2'],
                'rougeL': self.metrics['fine_tuned']['rouge_scores']['rougeL'] - self.metrics['baseline']['rouge_scores']['rougeL']
            }
        }

        return comparison

    def save_evaluation_results(self, output_file: str = "evaluation_results.json"):
        """Save evaluation results to file"""
        with open(output_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        logger.info(f"Evaluation results saved to {output_file}")

if __name__ == "__main__":
    # Test the fine-tuning system
    from data_processor import FinancialDataProcessor

    # Process documents and get Q&A pairs
    processor = FinancialDataProcessor()
    processed_texts, qa_pairs = processor.process_all_documents()

    # Initialize fine-tuning system
    fine_tune_system = FineTunedSystem()

    # Fine-tune on the data
    config = TrainingConfig(
        learning_rate=5e-5,
        batch_size=2,
        num_epochs=2,
        max_length=512,
        warmup_steps=50,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        save_steps=100,
        eval_steps=100,
        logging_steps=50
    )

    output_dir = fine_tune_system.fine_tune_on_data(qa_pairs, config)

    # Test questions
    test_questions = [
        "What was the company's revenue in 2024?",
        "What are the total assets?",
        "What type of company is this?"
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        response = fine_tune_system.answer_question(question)
        print(f"Answer: {response['answer']}")
        print(f"Confidence: {response['confidence']:.3f}")
        print(f"Response Time: {response['response_time']:.3f}s")
