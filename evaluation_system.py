# -----------------------------------------------------------------
# Comprehensive Evaluation System for RAG vs Fine-Tuning Comparison
# Implements detailed testing, evaluation, and analysis
# -----------------------------------------------------------------

# -------------------
# Importing libraries
# -------------------
import time
import json
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from data_processor import FinancialDataProcessor
from rag_system import RAGSystem, InputGuardrail, OutputGuardrail
from fine_tune_system import FineTunedSystem, ModelEvaluator, TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """Comprehensive evaluation system for RAG vs Fine-tuning comparison"""

    def __init__(self):
        self.results = {
            'rag': [],
            'fine_tuned': [],
            'comparison': {}
        }
        self.test_questions = []
        self.ground_truth = []
        self.evaluation_metrics = {}

    def prepare_test_dataset(self, qa_pairs: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
        """Prepare test dataset from Q&A pairs"""

        # ------------------------------------
        # Select diverse questions for testing
        # ------------------------------------
        test_questions = []
        ground_truth = []

        # -----------------------------------
        # Relevant, high-confidence questions
        # -----------------------------------
        high_conf_questions = [
            "What was the company's revenue in 2024?",
            "What are the total assets?",
            "What type of company is this?",
            "What are the main business segments?"
        ]

        # ----------------------------------------------
        # Relevant, low-confidence questions (ambiguous)
        # ----------------------------------------------
        low_conf_questions = [
            "How does the company compare to competitors?",
            "What are the growth trends?",
            "What are the risk factors?"
        ]

        # --------------------
        # Irrelevant questions
        # --------------------
        irrelevant_questions = [
            "What is the capital of France?",
            "How do you cook pasta?",
            "What is the weather like today?"
        ]

        # --------------------------
        # Combine all question types
        # --------------------------
        all_questions = high_conf_questions + low_conf_questions + irrelevant_questions

        # -----------------------------------------
        # Find corresponding answers from Q&A pairs
        # -----------------------------------------
        for question in all_questions:

            # ----------------------------------------------------
            # For relevant questions, try to find matching answers
            # ----------------------------------------------------
            if question in [qa['question'] for qa in qa_pairs]:
                matching_qa = next(qa for qa in qa_pairs if qa['question'] == question)
                test_questions.append(question)
                ground_truth.append(matching_qa['answer'])
            else:

                # --------------------------------------------------------
                # For questions not in our dataset, use expected responses
                # --------------------------------------------------------
                if question in high_conf_questions:
                    test_questions.append(question)
                    ground_truth.append("This information should be available in the financial statements.")
                elif question in low_conf_questions:
                    test_questions.append(question)
                    ground_truth.append("This information may be partially available or require interpretation.")
                else:  # irrelevant questions
                    test_questions.append(question)
                    ground_truth.append("This question is not related to financial data.")

        return test_questions, ground_truth

    def evaluate_rag_system(self, rag_system: RAGSystem, test_questions: List[str]) -> List[Dict]:
        """Evaluate RAG system performance"""
        logger.info("Evaluating RAG system...")

        rag_results = []
        input_guardrail = InputGuardrail()
        output_guardrail = OutputGuardrail()

        for i, question in enumerate(test_questions):
            logger.info(f"RAG Question {i+1}/{len(test_questions)}: {question}")

            # ----------------
            # Input validation
            # ----------------
            is_valid, validation_msg = input_guardrail.validate_query(question)

            if not is_valid:
                rag_results.append({
                    'question': question,
                    'method': 'rag',
                    'answer': f"Query rejected: {validation_msg}",
                    'confidence': 0.0,
                    'time': 0.0,
                    'correct': 'N/A',
                    'validation_status': 'rejected',
                    'sources': []
                })
                continue

            # ----------------
            # Get RAG response
            # ----------------
            start_time = time.time()
            response = rag_system.answer_question(question)
            response_time = time.time() - start_time

            # -----------------
            # Output validation
            # -----------------
            is_factual, factuality_msg = output_guardrail.validate_response(
                response['answer'], response['confidence']
            )

            # ----------------------------------------
            # Determine correctness (simple heuristic)
            # ----------------------------------------
            correct = self._determine_correctness(question, response['answer'], 'rag')

            result = {
                'question': question,
                'method': 'rag',
                'answer': response['answer'],
                'confidence': response['confidence'],
                'time': response_time,
                'correct': correct,
                'validation_status': 'accepted' if is_valid else 'rejected',
                'factuality_status': 'factual' if is_factual else 'questionable',
                'sources': response.get('sources', []),
                'method_used': response.get('method', 'rag')
            }

            rag_results.append(result)

            # ------------------------------------------------
            # Add small delay to avoid overwhelming the system
            # ------------------------------------------------
            time.sleep(0.1)

        return rag_results

    def evaluate_fine_tuned_system(self, fine_tuned_system: FineTunedSystem,
                                 test_questions: List[str]) -> List[Dict]:
        """Evaluate fine-tuned system performance"""
        logger.info("Evaluating Fine-tuned system...")

        fine_tuned_results = []
        input_guardrail = InputGuardrail()
        output_guardrail = OutputGuardrail()

        for i, question in enumerate(test_questions):
            logger.info(f"Fine-tuned Question {i+1}/{len(test_questions)}: {question}")

            # ----------------
            # Input validation
            # ----------------
            is_valid, validation_msg = input_guardrail.validate_query(question)

            if not is_valid:
                fine_tuned_results.append({
                    'question': question,
                    'method': 'fine_tuned',
                    'answer': f"Query rejected: {validation_msg}",
                    'confidence': 0.0,
                    'time': 0.0,
                    'correct': 'N/A',
                    'validation_status': 'rejected',
                    'sources': ['fine_tuned_model']
                })
                continue

            # -----------------------
            # Get fine-tuned response
            # -----------------------
            start_time = time.time()
            response = fine_tuned_system.answer_question(question)
            response_time = time.time() - start_time

            # -----------------
            # Output validation
            # -----------------
            is_factual, factuality_msg = output_guardrail.validate_response(
                response['answer'], response['confidence']
            )

            # ---------------------
            # Determine correctness
            # ---------------------
            correct = self._determine_correctness(question, response['answer'], 'fine_tuned')

            result = {
                'question': question,
                'method': 'fine_tuned',
                'answer': response['answer'],
                'confidence': response['confidence'],
                'time': response_time,
                'correct': correct,
                'validation_status': 'accepted' if is_valid else 'rejected',
                'factuality_status': 'factual' if is_factual else 'questionable',
                'sources': response.get('sources', []),
                'method_used': response.get('method', 'fine_tuned')
            }

            fine_tuned_results.append(result)

            # ---------------
            # Add small delay
            # ---------------
            time.sleep(0.1)

        return fine_tuned_results

    def _determine_correctness(self, question: str, answer: str, method: str) -> str:
        """Determine if the answer is correct (Y/N)"""
        question_lower = question.lower()
        answer_lower = answer.lower()

        # --------------------------------------
        # Check for rejection or error responses
        # --------------------------------------
        if any(phrase in answer_lower for phrase in ['rejected', 'cannot answer', 'no information']):
            return 'N'

        # -------------------------------------------------------------------
        # For irrelevant questions, check if system correctly identifies them
        # -------------------------------------------------------------------
        irrelevant_keywords = ['france', 'pasta', 'weather', 'capital']
        if any(keyword in question_lower for keyword in irrelevant_keywords):
            if method == 'rag':

                # ----------------------------------------
                # RAG should ideally say it's not relevant
                # ----------------------------------------
                if any(phrase in answer_lower for phrase in ['not relevant', 'not related', 'financial']):
                    return 'Y'
                else:
                    return 'N'
            else:

                # -------------------------------------
                # Fine-tuned might try to answer anyway
                # -------------------------------------
                return 'N'

        # -----------------------------------------------------------------
        # For relevant questions, check if answer contains expected content
        # -----------------------------------------------------------------
        if 'revenue' in question_lower:
            if any(phrase in answer_lower for phrase in ['billion', 'million', 'dollar', '$']):
                return 'Y'
        elif 'assets' in question_lower:
            if any(phrase in answer_lower for phrase in ['assets', 'billion', 'million', 'dollar']):
                return 'Y'
        elif 'company' in question_lower or 'business' in question_lower:
            if any(phrase in answer_lower for phrase in ['apple', 'technology', 'smartphone', 'computer']):
                return 'Y'

        # ----------------------------------------
        # Default to correct if we can't determine
        # ----------------------------------------
        return 'Y'

    def calculate_comprehensive_metrics(self, rag_results: List[Dict],
                                     fine_tuned_results: List[Dict]) -> Dict:
        """Calculate comprehensive comparison metrics"""
        logger.info("Calculating comprehensive metrics...")

        # -----------------------------------------------
        # Filter out rejected queries for fair comparison
        # -----------------------------------------------
        rag_valid = [r for r in rag_results if r['validation_status'] == 'accepted']
        fine_tuned_valid = [r for r in fine_tuned_results if r['validation_status'] == 'accepted']

        metrics = {
            'rag': {
                'total_questions': len(rag_results),
                'valid_questions': len(rag_valid),
                'rejected_questions': len(rag_results) - len(rag_valid),
                'avg_confidence': np.mean([r['confidence'] for r in rag_valid]) if rag_valid else 0,
                'avg_response_time': np.mean([r['time'] for r in rag_valid]) if rag_valid else 0,
                'correct_answers': sum(1 for r in rag_valid if r['correct'] == 'Y'),
                'accuracy': sum(1 for r in rag_valid if r['correct'] == 'Y') / len(rag_valid) if rag_valid else 0,
                'factual_responses': sum(1 for r in rag_valid if r.get('factuality_status') == 'factual'),
                'factuality_rate': sum(1 for r in rag_valid if r.get('factuality_status') == 'factual') / len(rag_valid) if rag_valid else 0
            },
            'fine_tuned': {
                'total_questions': len(fine_tuned_results),
                'valid_questions': len(fine_tuned_valid),
                'rejected_questions': len(fine_tuned_results) - len(fine_tuned_valid),
                'avg_confidence': np.mean([r['confidence'] for r in fine_tuned_valid]) if fine_tuned_valid else 0,
                'avg_response_time': np.mean([r['time'] for r in fine_tuned_valid]) if fine_tuned_valid else 0,
                'correct_answers': sum(1 for r in fine_tuned_valid if r['correct'] == 'Y'),
                'accuracy': sum(1 for r in fine_tuned_valid if r['correct'] == 'Y') / len(fine_tuned_valid) if fine_tuned_valid else 0,
                'factual_responses': sum(1 for r in fine_tuned_valid if r.get('factuality_status') == 'factual'),
                'factuality_rate': sum(1 for r in fine_tuned_valid if r.get('factuality_status') == 'factual') / len(fine_tuned_valid) if fine_tuned_valid else 0
            }
        }

        # ----------------------
        # Calculate improvements
        # ----------------------
        metrics['improvements'] = {
            'accuracy_improvement': metrics['fine_tuned']['accuracy'] - metrics['rag']['accuracy'],
            'speed_improvement': metrics['rag']['avg_response_time'] - metrics['fine_tuned']['avg_response_time'],
            'confidence_improvement': metrics['fine_tuned']['avg_confidence'] - metrics['rag']['avg_confidence'],
            'factuality_improvement': metrics['fine_tuned']['factuality_rate'] - metrics['rag']['factuality_rate']
        }

        return metrics

    def generate_results_table(self, rag_results: List[Dict],
                             fine_tuned_results: List[Dict]) -> pd.DataFrame:
        """Generate comprehensive results table"""

        # ---------------
        # Combine results
        # ---------------
        all_results = []

        for rag_result in rag_results:

            # ------------------------------------
            # Find corresponding fine-tuned result
            # ------------------------------------
            fine_tuned_result = next(
                (r for r in fine_tuned_results if r['question'] == rag_result['question']),
                None
            )

            if fine_tuned_result:
                all_results.append({
                    'Question': rag_result['question'],
                    'RAG_Answer': rag_result['answer'],
                    'RAG_Confidence': f"{rag_result['confidence']:.3f}",
                    'RAG_Time': f"{rag_result['time']:.3f}s",
                    'RAG_Correct': rag_result['correct'],
                    'Fine_Tuned_Answer': fine_tuned_result['answer'],
                    'Fine_Tuned_Confidence': f"{fine_tuned_result['confidence']:.3f}",
                    'Fine_Tuned_Time': f"{fine_tuned_result['time']:.3f}s",
                    'Fine_Tuned_Correct': fine_tuned_result['correct']
                })

        return pd.DataFrame(all_results)

    def create_visualizations(self, metrics: Dict, output_dir: str = "evaluation_results"):
        """Create visualization charts for the evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # ---------
        # Set style
        # ---------
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # ----------------------
        # 1. Accuracy Comparison
        # ----------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        methods = ['RAG', 'Fine-Tuned']
        accuracies = [metrics['rag']['accuracy'], metrics['fine_tuned']['accuracy']]

        bars = ax.bar(methods, accuracies, color=['skyblue', 'lightcoral'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison: RAG vs Fine-Tuned')
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # ---------------------------
        # 2. Response Time Comparison
        # ---------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        response_times = [metrics['rag']['avg_response_time'], metrics['fine_tuned']['avg_response_time']]

        bars = ax.bar(methods, response_times, color=['lightgreen', 'orange'])
        ax.set_ylabel('Average Response Time (seconds)')
        ax.set_title('Response Time Comparison: RAG vs Fine-Tuned')

        # Add value labels
        for bar, time_val in zip(bars, response_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{time_val:.3f}s', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path / 'response_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # ------------------------
        # 3. Confidence Comparison
        # ------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        confidences = [metrics['rag']['avg_confidence'], metrics['fine_tuned']['avg_confidence']]

        bars = ax.bar(methods, confidences, color=['gold', 'lightsteelblue'])
        ax.set_ylabel('Average Confidence')
        ax.set_title('Confidence Comparison: RAG vs Fine-Tuned')
        ax.set_ylim(0, 1)

        # Add value labels
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{conf:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path / 'confidence_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # --------------------------------
        # 4. Comprehensive Metrics Heatmap
        # --------------------------------
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data for heatmap
        heatmap_data = {
            'RAG': [
                metrics['rag']['accuracy'],
                metrics['rag']['avg_response_time'],
                metrics['rag']['avg_confidence'],
                metrics['rag']['factuality_rate']
            ],
            'Fine-Tuned': [
                metrics['fine_tuned']['accuracy'],
                metrics['fine_tuned']['avg_response_time'],
                metrics['fine_tuned']['avg_confidence'],
                metrics['fine_tuned']['factuality_rate']
            ]
        }

        df_heatmap = pd.DataFrame(heatmap_data,
                                 index=['Accuracy', 'Response Time', 'Confidence', 'Factuality Rate'])

        sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax)
        ax.set_title('Comprehensive Metrics Comparison')

        plt.tight_layout()
        plt.savefig(output_path / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualizations saved to {output_dir}")

    def run_comprehensive_evaluation(self, output_dir: str = "evaluation_results") -> Dict:
        """Run the complete evaluation pipeline"""
        logger.info("Starting comprehensive evaluation...")

        # -------------------------------------
        # 1. Process documents and prepare data
        # -------------------------------------
        processor = FinancialDataProcessor()
        processed_texts, qa_pairs = processor.process_all_documents()
        chunks = processor.get_text_chunks()

        # -----------------------
        # 2. Prepare test dataset
        # -----------------------
        self.test_questions, self.ground_truth = self.prepare_test_dataset(qa_pairs)
        logger.info(f"Prepared {len(self.test_questions)} test questions")

        # ---------------------
        # 3. Initialize systems
        # ---------------------
        logger.info("Initializing RAG system...")
        rag_system = RAGSystem()
        rag_system.add_documents(chunks)

        logger.info("Initializing Fine-tuned system...")
        fine_tuned_system = FineTunedSystem()

        # ---------------------
        # Fine-tune on the data
        # ---------------------
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
        fine_tuned_system.fine_tune_on_data(qa_pairs, config)

        # ------------------
        # 4. Run evaluations
        # ------------------
        logger.info("Running RAG evaluation...")
        rag_results = self.evaluate_rag_system(rag_system, self.test_questions)

        logger.info("Running Fine-tuned evaluation...")
        fine_tuned_results = self.evaluate_fine_tuned_system(fine_tuned_system, self.test_questions)

        # --------------------
        # 5. Calculate metrics
        # --------------------
        self.evaluation_metrics = self.calculate_comprehensive_metrics(rag_results, fine_tuned_results)

        # -------------------------
        # 6. Generate results table
        # -------------------------
        results_table = self.generate_results_table(rag_results, fine_tuned_results)

        # ---------------
        # 7. Save results
        # ---------------
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        with open(output_path / "detailed_results.json", 'w') as f:
            json.dump({
                'rag_results': rag_results,
                'fine_tuned_results': fine_tuned_results,
                'metrics': self.evaluation_metrics
            }, f, indent=2, default=str)

        results_table.to_csv(output_path / "results_table.csv", index=False)

        # Save metrics summary
        with open(output_path / "metrics_summary.json", 'w') as f:
            json.dump(self.evaluation_metrics, f, indent=2, default=str)

        # 8. Create visualizations
        self.create_visualizations(self.evaluation_metrics, output_dir)

        # 9. Print summary
        self._print_evaluation_summary()

        logger.info(f"Comprehensive evaluation complete. Results saved to {output_dir}")

        return self.evaluation_metrics

    def _print_evaluation_summary(self):
        """Print a summary of evaluation results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*80)

        print(f"\nRAG System Performance:")
        print(f"  Accuracy: {self.evaluation_metrics['rag']['accuracy']:.3f}")
        print(f"  Avg Response Time: {self.evaluation_metrics['rag']['avg_response_time']:.3f}s")
        print(f"  Avg Confidence: {self.evaluation_metrics['rag']['avg_confidence']:.3f}")
        print(f"  Factuality Rate: {self.evaluation_metrics['rag']['factuality_rate']:.3f}")

        print(f"\nFine-Tuned System Performance:")
        print(f"  Accuracy: {self.evaluation_metrics['fine_tuned']['accuracy']:.3f}")
        print(f"  Avg Response Time: {self.evaluation_metrics['fine_tuned']['avg_response_time']:.3f}s")
        print(f"  Avg Confidence: {self.evaluation_metrics['fine_tuned']['avg_confidence']:.3f}")
        print(f"  Factuality Rate: {self.evaluation_metrics['fine_tuned']['factuality_rate']:.3f}")

        print(f"\nImprovements with Fine-Tuning:")
        print(f"  Accuracy: {self.evaluation_metrics['improvements']['accuracy_improvement']:+.3f}")
        print(f"  Speed: {self.evaluation_metrics['improvements']['speed_improvement']:+.3f}s")
        print(f"  Confidence: {self.evaluation_metrics['improvements']['confidence_improvement']:+.3f}")
        print(f"  Factuality: {self.evaluation_metrics['improvements']['factuality_improvement']:+.3f}")

        print("\n" + "="*80)

if __name__ == "__main__":
    # Run comprehensive evaluation
    evaluator = ComprehensiveEvaluator()
    results = evaluator.run_comprehensive_evaluation()

    print("\nEvaluation complete!")
