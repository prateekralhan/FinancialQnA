#!/usr/bin/env python3
# ---------------------------------------------------------------
# Main Execution Script for Financial QA System
# Provides command-line interface for different system components
# ---------------------------------------------------------------

import sys
import logging
import argparse
from pathlib import Path


# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_system import RAGSystem # type: ignore
from data_processor import FinancialDataProcessor # type: ignore
from evaluation_system import ComprehensiveEvaluator # type: ignore
from fine_tune_system import FineTunedSystem, TrainingConfig # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_data_processing():
    """Run data processing pipeline"""
    logger.info("Starting data processing...")

    processor = FinancialDataProcessor()
    processed_texts, qa_pairs = processor.process_all_documents()
    processor.save_processed_data()

    # Generate chunks
    chunks = processor.get_text_chunks()

    logger.info(f"Data processing complete!")
    logger.info(f"Processed {len(processed_texts)} documents")
    logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
    logger.info(f"Created {len(chunks)} text chunks")

    return processed_texts, qa_pairs, chunks

def run_rag_system(chunks):
    """Run RAG system evaluation"""
    logger.info("Initializing RAG system...")

    rag_system = RAGSystem()
    rag_system.add_documents(chunks)

    # Test questions
    test_questions = [
        "What was the company's revenue in 2024?",
        "What are the total assets?",
        "What type of company is this?",
        "What is the capital of France?"  # Irrelevant question
    ]

    logger.info("Testing RAG system...")
    for question in test_questions:
        logger.info(f"\nQuestion: {question}")
        response = rag_system.answer_question(question)
        logger.info(f"Answer: {response['answer']}")
        logger.info(f"Confidence: {response['confidence']:.3f}")
        logger.info(f"Method: {response['method']}")
        logger.info(f"Response Time: {response['response_time']:.3f}s")

    return rag_system

def run_fine_tuned_system(qa_pairs):
    """Run fine-tuned system evaluation"""
    logger.info("Initializing Fine-tuned system...")

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

    logger.info("Starting fine-tuning...")
    output_dir = fine_tune_system.fine_tune_on_data(qa_pairs, config)
    logger.info(f"Fine-tuning complete. Model saved to {output_dir}")

    # Test questions
    test_questions = [
        "What was the company's revenue in 2024?",
        "What are the total assets?",
        "What type of company is this?"
    ]

    logger.info("Testing Fine-tuned system...")
    for question in test_questions:
        logger.info(f"\nQuestion: {question}")
        response = fine_tune_system.answer_question(question)
        logger.info(f"Answer: {response['answer']}")
        logger.info(f"Confidence: {response['confidence']:.3f}")
        logger.info(f"Response Time: {response['response_time']:.3f}s")

    return fine_tune_system

def run_comprehensive_evaluation():
    """Run comprehensive evaluation"""
    logger.info("Starting comprehensive evaluation...")

    evaluator = ComprehensiveEvaluator()
    results = evaluator.run_comprehensive_evaluation()

    logger.info("Comprehensive evaluation complete!")
    return results

def run_streamlit_interface():
    """Run Streamlit interface"""
    logger.info("Starting Streamlit interface...")

    import subprocess
    import os

    # Change to src directory
    os.chdir(Path(__file__).parent / "src")

    # Run streamlit
    cmd = ["streamlit", "run", "interface.py"]
    logger.info(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Streamlit failed: {e}")
    except FileNotFoundError:
        logger.error("Streamlit not found. Please install with: pip install streamlit")

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Financial QA System: RAG vs Fine-tuning Comparison"
    )

    parser.add_argument(
        "mode",
        choices=["data", "rag", "fine-tune", "evaluate", "interface", "all"],
        help="Mode to run"
    )

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for results"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    try:
        if args.mode == "data":
            run_data_processing()

        elif args.mode == "rag":
            # First process data, then run RAG
            _, _, chunks = run_data_processing()
            run_rag_system(chunks)

        elif args.mode == "fine-tune":
            # First process data, then run fine-tuning
            _, qa_pairs, _ = run_data_processing()
            run_fine_tuned_system(qa_pairs)

        elif args.mode == "evaluate":
            run_comprehensive_evaluation()

        elif args.mode == "interface":
            run_streamlit_interface()

        elif args.mode == "all":
            # Run complete pipeline
            logger.info("Running complete pipeline...")

            # 1. Data processing
            processed_texts, qa_pairs, chunks = run_data_processing()

            # 2. RAG system
            rag_system = run_rag_system(chunks)

            # 3. Fine-tuned system
            fine_tuned_system = run_fine_tuned_system(qa_pairs)

            # 4. Comprehensive evaluation
            results = run_comprehensive_evaluation()

            logger.info("Complete pipeline finished successfully!")

        logger.info(f"Mode '{args.mode}' completed successfully!")

    except Exception as e:
        logger.error(f"Error in mode '{args.mode}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
