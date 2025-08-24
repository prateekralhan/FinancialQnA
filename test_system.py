#!/usr/bin/env python3
# -------------------------------------------
# Test Script for Financial QA System
# Verifies that all components work correctly
# -------------------------------------------

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_processor():
    """Test the data processor"""
    logger.info("Testing Data Processor...")

    try:
        from data_processor import FinancialDataProcessor

        processor = FinancialDataProcessor()
        processed_texts, qa_pairs = processor.process_all_documents()

        logger.info(f"✅ Data processing successful!")
        logger.info(f"   - Processed {len(processed_texts)} documents")
        logger.info(f"   - Generated {len(qa_pairs)} Q&A pairs")

        # Test chunking
        chunks = processor.get_text_chunks()
        logger.info(f"   - Created {len(chunks)} text chunks")

        return True, processed_texts, qa_pairs, chunks

    except Exception as e:
        logger.error(f"❌ Data processor test failed: {e}")
        return False, None, None, None

def test_rag_system(chunks):
    """Test the RAG system"""
    logger.info("Testing RAG System...")

    try:
        from rag_system import RAGSystem

        rag_system = RAGSystem()
        rag_system.add_documents(chunks)

        # Test a simple question
        test_question = "What type of company is this?"
        response = rag_system.answer_question(test_question)

        logger.info(f"✅ RAG system test successful!")
        logger.info(f"   - Question: {test_question}")
        logger.info(f"   - Answer: {response['answer'][:100]}...")
        logger.info(f"   - Confidence: {response['confidence']:.3f}")

        return True, rag_system

    except Exception as e:
        logger.error(f"❌ RAG system test failed: {e}")
        return False, None

def test_fine_tuned_system(qa_pairs):
    """Test the fine-tuned system"""
    logger.info("Testing Fine-tuned System...")

    try:
        from fine_tune_system import FineTunedSystem, TrainingConfig

        fine_tune_system = FineTunedSystem()

        # Fine-tune on a small subset for testing
        test_qa_pairs = qa_pairs[:5]  # Use only 5 pairs for quick testing

        config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=1,  # Small batch size for testing
            num_epochs=1,   # Single epoch for testing
            max_length=512,
            warmup_steps=10,
            weight_decay=0.01,
            gradient_accumulation_steps=1,
            save_steps=50,
            eval_steps=50,
            logging_steps=25
        )

        logger.info("Starting fine-tuning (this may take a few minutes)...")
        output_dir = fine_tune_system.fine_tune_on_data(test_qa_pairs, config)

        # Test a simple question
        test_question = "What type of company is this?"
        response = fine_tune_system.answer_question(test_question)

        logger.info(f"✅ Fine-tuned system test successful!")
        logger.info(f"   - Model saved to: {output_dir}")
        logger.info(f"   - Question: {test_question}")
        logger.info(f"   - Answer: {response['answer'][:100]}...")
        logger.info(f"   - Confidence: {response['confidence']:.3f}")

        return True, fine_tune_system

    except Exception as e:
        logger.error(f"❌ Fine-tuned system test failed: {e}")
        return False, None

def test_evaluation_system():
    """Test the evaluation system"""
    logger.info("Testing Evaluation System...")

    try:
        from evaluation_system import ComprehensiveEvaluator

        evaluator = ComprehensiveEvaluator()

        logger.info("✅ Evaluation system test successful!")
        logger.info("   - System components loaded correctly")

        return True, evaluator

    except Exception as e:
        logger.error(f"❌ Evaluation system test failed: {e}")
        return False, None

def main():
    """Run all tests"""
    logger.info("🚀 Starting Financial QA System Tests...")
    logger.info("=" * 50)

    # Test 1: Data Processor
    success, processed_texts, qa_pairs, chunks = test_data_processor()
    if not success:
        logger.error("❌ System test failed at data processing stage")
        return False

    # Test 2: RAG System
    success, rag_system = test_rag_system(chunks)
    if not success:
        logger.error("❌ System test failed at RAG system stage")
        return False

    # Test 3: Fine-tuned System
    success, fine_tuned_system = test_fine_tuned_system(qa_pairs)
    if not success:
        logger.error("❌ System test failed at fine-tuned system stage")
        return False

    # Test 4: Evaluation System
    success, evaluator = test_evaluation_system()
    if not success:
        logger.error("❌ System test failed at evaluation system stage")
        return False

    logger.info("=" * 50)
    logger.info("🎉 All tests passed successfully!")
    logger.info("✅ Financial QA System is ready to use!")

    # Print summary
    logger.info("\n📊 System Summary:")
    logger.info(f"   - Documents processed: {len(processed_texts)}")
    logger.info(f"   - Q&A pairs generated: {len(qa_pairs)}")
    logger.info(f"   - Text chunks created: {len(chunks)}")
    logger.info(f"   - RAG system: Ready")
    logger.info(f"   - Fine-tuned system: Ready")
    logger.info(f"   - Evaluation system: Ready")

    logger.info("\n🚀 Next steps:")
    logger.info("   1. Run 'python main.py interface' for web interface")
    logger.info("   2. Run 'python main.py evaluate' for comprehensive evaluation")
    logger.info("   3. Run 'python main.py all' for complete pipeline")

    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
