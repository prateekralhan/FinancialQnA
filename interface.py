# ------------------------------------------------------------------
# Streamlit based User Interface for Financial QA System
# Provides unified interface for both RAG and Fine-tuned approaches
# ------------------------------------------------------------------

# -------------------
# Importing libraries
# -------------------
import time
import json
import logging
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict, List
from data_processor import FinancialDataProcessor
from evaluation_system import ComprehensiveEvaluator
from fine_tune_system import FineTunedSystem, TrainingConfig
from rag_system import RAGSystem, InputGuardrail, OutputGuardrail
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialQAInterface:
    """Streamlit interface for Financial QA System"""

    def __init__(self, test_mode=False):
        self.rag_system = None
        self.fine_tuned_system = None
        self.processor = None
        self.qa_pairs = []
        self.chunks = []
        self.test_mode = test_mode

        # -----------------------------------------
        # Initialize systems (skip if in test mode)
        # -----------------------------------------
        if not test_mode:
            self._initialize_systems()

    def _initialize_systems(self):
        """Initialize RAG and Fine-tuned systems"""
        try:

            # -----------------------------------
            # Check if required directories exist
            # -----------------------------------
            st.info("🔍 Checking system requirements...")
            required_dirs = ["data", "models", "evaluation_results"]
            for dir_name in required_dirs:
                dir_path = Path(dir_name)
                if not dir_path.exists():
                    st.warning(f"⚠️ Directory '{dir_name}' not found. Creating it...")
                    dir_path.mkdir(exist_ok=True)
                    st.success(f"✅ Created directory '{dir_name}'")

            # -----------------
            # Process documents
            # -----------------
            st.info("🔄 Processing documents...")
            try:
                self.processor = FinancialDataProcessor()
                processed_texts, self.qa_pairs = self.processor.process_all_documents()
                self.chunks = self.processor.get_text_chunks()

                if not self.qa_pairs or len(self.qa_pairs) == 0:
                    st.warning("⚠️ No Q&A pairs found. Please check your document processing.")
                    return

                if not self.chunks or len(self.chunks) == 0:
                    st.warning("⚠️ No text chunks found. Please check your document processing.")
                    return

                st.success(f"✅ Processed {len(self.qa_pairs)} Q&A pairs and {len(self.chunks)} text chunks")

            except Exception as e:
                st.error(f"❌ Failed to process documents: {e}")
                logger.error(f"Document processing error: {e}")
                raise

            # --------------
            # Initialize RAG
            # --------------
            with st.spinner("Initializing RAG system..."):
                try:
                    self.rag_system = RAGSystem()
                    self.rag_system.add_documents(self.chunks)
                    st.success("✅ RAG System initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize RAG system: {e}")
                    logger.error(f"RAG system initialization error: {e}")
                    raise

            # ----------------------------
            # Initialize Fine-tuned system
            # ----------------------------
            with st.spinner("Initializing Fine-tuned system..."):
                try:
                    self.fine_tuned_system = FineTunedSystem()

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
                    self.fine_tuned_system.fine_tune_on_data(self.qa_pairs, config)
                    st.success("✅ Fine-tuned System initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize Fine-tuned system: {e}")
                    logger.error(f"Fine-tuned system initialization error: {e}")
                    raise

            st.success("✅ All systems initialized successfully!")

        except Exception as e:
            st.error(f"❌ Error initializing systems: {e}")
            logger.error(f"Initialization error: {e}")
            # Set systems to None to prevent further errors
            self.rag_system = None
            self.fine_tuned_system = None
            # Don't raise here, let the interface handle it gracefully

    def run(self):
        """Run the Streamlit interface"""
        st.set_page_config(
            page_title="Financial QA System - RAG vs Fine-tuning",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("📊 Financial QA System: RAG vs Fine-tuning Comparison")
        st.markdown("---")

        # ----------------
        # Handle test mode
        # ----------------
        if self.test_mode:
            st.warning("🧪 **Test Mode Active** - Systems are not initialized. Use this mode to test the interface layout.")
            st.info("To run the full system, uncheck 'Test Mode' in the sidebar and restart the app.")

            # Create a simple test interface
            self._create_test_interface()
            return

        # --------------------------------
        # Check if systems are initialized
        # --------------------------------
        if not hasattr(self, 'rag_system') or not self.rag_system:
            st.error("❌ RAG System failed to initialize. Please check the logs above.")
            st.stop()

        if not hasattr(self, 'fine_tuned_system') or not self.fine_tuned_system:
            st.error("❌ Fine-tuned System failed to initialize. Please check the logs above.")
            st.stop()

        # -------
        # Sidebar
        # -------
        system_mode = self._create_sidebar()

        # ------------
        # Main content
        # ------------
        self._create_main_content(system_mode)

    def _create_sidebar(self):
        """Create the sidebar with system information and controls"""
        st.sidebar.header("🔧 System Controls")

        # ----------------
        # System selection
        # ----------------
        st.sidebar.subheader("Select System")
        system_mode = st.sidebar.selectbox(
            "Choose QA System:",
            ["RAG System", "Fine-tuned System", "Both (Comparison)"],
            index=0,
            key="system_mode_select"
        )

        # -------------------
        # Display system info
        # -------------------
        st.sidebar.subheader("📋 System Information")

        if hasattr(self, 'qa_pairs') and self.qa_pairs:
            st.sidebar.metric("Q&A Pairs", len(self.qa_pairs))
        else:
            st.sidebar.metric("Q&A Pairs", "N/A")

        if hasattr(self, 'chunks') and self.chunks:
            st.sidebar.metric("Text Chunks", len(self.chunks))
        else:
            st.sidebar.metric("Text Chunks", "N/A")

        # -------------
        # System status
        # -------------
        st.sidebar.subheader("🟢 System Status")
        if hasattr(self, 'rag_system') and self.rag_system:
            st.sidebar.success("RAG System: Ready")
        else:
            st.sidebar.error("RAG System: Not Ready")

        if hasattr(self, 'fine_tuned_system') and self.fine_tuned_system:
            st.sidebar.success("Fine-tuned System: Ready")
        else:
            st.sidebar.error("Fine-tuned System: Not Ready")

        # ----------------
        # Advanced options
        # ----------------
        st.sidebar.subheader("⚙️ Advanced Options")

        # --------------
        # RAG parameters
        # --------------
        if system_mode in ["RAG System", "Both (Comparison)"]:
            st.sidebar.number_input(
                "RAG Top-K Retrieval:",
                min_value=1,
                max_value=10,
                value=5,
                key="rag_top_k"
            )

        # ----------------------
        # Fine-tuning parameters
        # ----------------------
        if system_mode in ["Fine-tuned System", "Both (Comparison)"]:
            st.sidebar.slider(
                "Generation Temperature:",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="gen_temperature"
            )

        return system_mode

    def _create_main_content(self, system_mode: str):
        """Create the main content area"""

        # ---------
        # Main tabs
        # ---------
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 Interactive QA",
            "📊 Evaluation Results",
            "📈 System Comparison",
            "📚 Documentation"
        ])

        with tab1:
            self._create_qa_interface(system_mode)

        with tab2:
            self._create_evaluation_interface()

        with tab3:
            self._create_comparison_interface()

        with tab4:
            self._create_documentation_interface()

    def _create_qa_interface(self, system_mode: str):
        """Create the interactive QA interface"""
        st.header("💬 Interactive Question & Answer")

        # --------------------------
        # Check if systems are ready
        # --------------------------
        if not hasattr(self, 'rag_system') or not self.rag_system:
            st.error("❌ RAG System is not initialized. Please check the initialization logs.")
            return

        if system_mode in ["Fine-tuned System", "Both (Comparison)"] and (not hasattr(self, 'fine_tuned_system') or not self.fine_tuned_system):
            st.error("❌ Fine-tuned System is not initialized. Please check the initialization logs.")
            return

        # --------------
        # Question input
        # --------------
        question = st.text_input(
            "Ask a financial question:",
            placeholder="e.g., What was the company's revenue in 2024?",
            key="user_question"
        )

        if st.button("🚀 Get Answer", key="get_answer_btn"):
            if question.strip():
                self._process_question(question, system_mode)
            else:
                st.warning("Please enter a question.")

    def _process_question(self, question: str, system_mode: str):
        """Process a question using the selected system(s)"""
        st.subheader("📝 Question")
        st.write(f"**Q:** {question}")

        # ----------------
        # Input validation
        # ----------------
        input_guardrail = InputGuardrail()
        is_valid, validation_msg = input_guardrail.validate_query(question)

        if not is_valid:
            st.error(f"❌ Query rejected: {validation_msg}")
            return

        st.success(f"✅ Query validated: {validation_msg}")

        # -------------------------------
        # Process with selected system(s)
        # -------------------------------
        if system_mode == "RAG System":
            self._process_with_rag(question)
        elif system_mode == "Fine-tuned System":
            self._process_with_fine_tuned(question)
        elif system_mode == "Both (Comparison)":
            self._process_with_both(question)

    def _process_with_rag(self, question: str):
        """Process question using RAG system"""
        st.subheader("🔍 RAG System Response")

        with st.spinner("Retrieving and generating answer..."):
            start_time = time.time()
            response = self.rag_system.answer_question(question)
            response_time = time.time() - start_time

        # ---------------
        # Display results
        # ---------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Confidence", f"{response['confidence']:.3f}")

        with col2:
            st.metric("Response Time", f"{response['response_time']:.3f}s")

        with col3:
            st.metric("Method", response.get('method', 'rag').upper())

        # ------
        # Answer
        # ------
        st.subheader("💡 Answer")
        st.write(response['answer'])

        # -------
        # Sources
        # -------
        if response.get('sources'):
            st.subheader("📚 Sources")
            for source in response['sources']:
                st.write(f"• {source}")

        # -----------------
        # Output validation
        # -----------------
        output_guardrail = OutputGuardrail()
        is_factual, factuality_msg = output_guardrail.validate_response(
            response['answer'], response['confidence']
        )

        if is_factual:
            st.success(f"✅ Response validation: {factuality_msg}")
        else:
            st.warning(f"⚠️ Response validation: {factuality_msg}")

    def _process_with_fine_tuned(self, question: str):
        """Process question using Fine-tuned system"""
        st.subheader("🎯 Fine-tuned System Response")

        with st.spinner("Generating answer with fine-tuned model..."):
            start_time = time.time()
            response = self.fine_tuned_system.answer_question(question)
            response_time = time.time() - start_time

        # ---------------
        # Display results
        # ---------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Confidence", f"{response['confidence']:.3f}")

        with col2:
            st.metric("Response Time", f"{response['response_time']:.3f}s")

        with col3:
            st.metric("Method", response.get('method', 'fine_tuned').upper())

        st.subheader("💡 Answer")
        st.write(response['answer'])

        # -----------------
        # Output validation
        # -----------------
        output_guardrail = OutputGuardrail()
        is_factual, factuality_msg = output_guardrail.validate_response(
            response['answer'], response['confidence']
        )

        if is_factual:
            st.success(f"✅ Response validation: {factuality_msg}")
        else:
            st.warning(f"⚠️ Response validation: {factuality_msg}")

    def _process_with_both(self, question: str):
        """Process question using both systems for comparison"""
        st.subheader("🔄 Dual System Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 RAG System")
            with st.spinner("RAG processing..."):
                rag_response = self.rag_system.answer_question(question)

            st.metric("Confidence", f"{rag_response['confidence']:.3f}")
            st.metric("Response Time", f"{rag_response['response_time']:.3f}s")
            st.write("**Answer:**", rag_response['answer'])

        with col2:
            st.subheader("🎯 Fine-tuned System")
            with st.spinner("Fine-tuned processing..."):
                ft_response = self.fine_tuned_system.answer_question(question)

            st.metric("Confidence", f"{ft_response['confidence']:.3f}")
            st.metric("Response Time", f"{ft_response['response_time']:.3f}s")
            st.write("**Answer:**", ft_response['answer'])

        # ------------------
        # Comparison metrics
        # ------------------
        st.subheader("📊 Quick Comparison")

        col1, col2, col3 = st.columns(3)

        with col1:
            confidence_diff = ft_response['confidence'] - rag_response['confidence']
            st.metric("Confidence Δ", f"{confidence_diff:+.3f}")

        with col2:
            time_diff = rag_response['response_time'] - ft_response['response_time']
            st.metric("Speed Δ", f"{time_diff:+.3f}s")

        with col3:
            if confidence_diff > 0 and time_diff > 0:
                st.success("Fine-tuned wins!")
            elif confidence_diff < 0 and time_diff < 0:
                st.info("RAG wins!")
            else:
                st.warning("Mixed results")

    def _create_evaluation_interface(self):
        """Create the evaluation interface"""
        st.header("📊 Evaluation Results")

        # --------------------------
        # Check if systems are ready
        # --------------------------
        if not hasattr(self, 'rag_system') or not self.rag_system:
            st.error("❌ RAG System is not initialized. Cannot run evaluation.")
            return

        if not hasattr(self, 'fine_tuned_system') or not self.fine_tuned_system:
            st.error("❌ Fine-tuned System is not initialized. Cannot run evaluation.")
            return

        if st.button("🚀 Run Comprehensive Evaluation", key="run_eval_btn"):
            with st.spinner("Running comprehensive evaluation..."):
                try:
                    evaluator = ComprehensiveEvaluator()
                    results = evaluator.run_comprehensive_evaluation()

                    st.success("Evaluation complete!")

                    # Display results
                    self._display_evaluation_results(results)
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {e}")
                    logger.error(f"Evaluation error: {e}")

    def _display_evaluation_results(self, results: Dict):
        """Display evaluation results"""
        st.subheader("📈 Performance Metrics")

        # ----------------------
        # Create metrics display
        # ----------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 RAG System")
            st.metric("Accuracy", f"{results['rag']['accuracy']:.3f}")
            st.metric("Avg Response Time", f"{results['rag']['avg_response_time']:.3f}s")
            st.metric("Avg Confidence", f"{results['rag']['avg_confidence']:.3f}")
            st.metric("Factuality Rate", f"{results['rag']['factuality_rate']:.3f}")

        with col2:
            st.subheader("🎯 Fine-tuned System")
            st.metric("Accuracy", f"{results['fine_tuned']['accuracy']:.3f}")
            st.metric("Avg Response Time", f"{results['fine_tuned']['avg_response_time']:.3f}s")
            st.metric("Avg Confidence", f"{results['fine_tuned']['avg_confidence']:.3f}")
            st.metric("Factuality Rate", f"{results['fine_tuned']['factuality_rate']:.3f}")

        # ------------
        # Improvements
        # ------------
        st.subheader("🚀 Improvements with Fine-tuning")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy Δ", f"{results['improvements']['accuracy_improvement']:+.3f}")

        with col2:
            st.metric("Speed Δ", f"{results['improvements']['speed_improvement']:+.3f}s")

        with col3:
            st.metric("Confidence Δ", f"{results['improvements']['confidence_improvement']:+.3f}")

        with col4:
            st.metric("Factuality Δ", f"{results['improvements']['factuality_improvement']:+.3f}")

    def _create_comparison_interface(self):
        """Create the system comparison interface"""
        st.header("📈 System Comparison")

        # --------------------------
        # Check if systems are ready
        # --------------------------
        if not hasattr(self, 'rag_system') or not self.rag_system:
            st.error("❌ RAG System is not initialized. Cannot display comparison.")
            return

        if not hasattr(self, 'fine_tuned_system') or not self.fine_tuned_system:
            st.error("❌ Fine-tuned System is not initialized. Cannot display comparison.")
            return

        # ------------------------------------
        # Load evaluation results if available
        # ------------------------------------
        eval_file = Path("evaluation_results/metrics_summary.json")
        if eval_file.exists():
            try:
                with open(eval_file, 'r') as f:
                    results = json.load(f)

                self._display_comparison_charts(results)
            except Exception as e:
                st.error(f"❌ Error loading evaluation results: {e}")
                logger.error(f"Error loading evaluation results: {e}")
        else:
            st.info("Run the evaluation first to see comparison charts.")

    def _display_comparison_charts(self, results: Dict):
        """Display comparison charts"""
        st.subheader("📊 Performance Comparison Charts")

        # -----------------------
        # Load and display charts
        # -----------------------
        chart_dir = Path("evaluation_results")

        if (chart_dir / "accuracy_comparison.png").exists():
            st.image(chart_dir / "accuracy_comparison.png", caption="Accuracy Comparison")

        if (chart_dir / "response_time_comparison.png").exists():
            st.image(chart_dir / "response_time_comparison.png", caption="Response Time Comparison")

        if (chart_dir / "confidence_comparison.png").exists():
            st.image(chart_dir / "confidence_comparison.png", caption="Confidence Comparison")

        if (chart_dir / "metrics_heatmap.png").exists():
            st.image(chart_dir / "metrics_heatmap.png", caption="Comprehensive Metrics Heatmap")

    def _create_documentation_interface(self):
        """Create the documentation interface"""
        st.header("📚 System Documentation")

        st.subheader("🔍 RAG System Features")
        st.markdown("""
        - **Hybrid Retrieval**: Combines dense (vector) and sparse (BM25) retrieval
        - **Memory-Augmented Retrieval**: Persistent memory bank for frequent Q&A
        - **Advanced Guardrails**: Input and output validation systems
        - **Multi-source Retrieval**: FAISS vector database + ChromaDB
        """)

        st.subheader("🎯 Fine-tuned System Features")
        st.markdown("""
        - **Continual Learning**: Incremental fine-tuning without catastrophic forgetting
        - **Domain Adaptation**: Specialized for financial Q&A
        - **Efficient Training**: Optimized hyperparameters for small models
        - **Confidence Scoring**: Built-in confidence estimation
        """)

        st.subheader("📊 Evaluation Metrics")
        st.markdown("""
        - **Accuracy**: Correct answer rate
        - **Response Time**: Average inference speed
        - **Confidence**: Model confidence scores
        - **Factuality**: Response reliability assessment
        - **ROUGE Scores**: Text similarity metrics
        """)

        st.subheader("🚀 Getting Started")
        st.markdown("""
        1. **Select System**: Choose between RAG, Fine-tuned, or both
        2. **Ask Questions**: Input financial questions in the QA interface
        3. **View Results**: Compare performance metrics and responses
        4. **Run Evaluation**: Execute comprehensive system comparison
        5. **Analyze Charts**: Review performance visualizations
        """)

    def _create_test_interface(self):
        """Create a simple test interface for testing the layout"""
        st.header("🧪 Test Interface")
        st.info("This is a test mode to verify the interface layout works correctly.")

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 Interactive QA",
            "📊 Evaluation Results",
            "📈 System Comparison",
            "📚 Documentation"
        ])

        with tab1:
            st.subheader("💬 Interactive Question & Answer")
            st.info("QA interface would be here in full mode")
            question = st.text_input("Test Question Input:", key="test_question")
            if st.button("Test Button", key="test_button"):
                st.success("✅ Test button works!")

        with tab2:
            st.subheader("📊 Evaluation Results")
            st.info("Evaluation interface would be here in full mode")

        with tab3:
            st.subheader("📈 System Comparison")
            st.info("Comparison interface would be here in full mode")

        with tab4:
            st.subheader("📚 Documentation")
            st.info("Documentation interface would be here in full mode")

def main():
    """Main function to run the interface"""
    try:
        st.info("🚀 Starting Financial QA System...")

        # Add a test mode option
        test_mode = st.sidebar.checkbox("🧪 Test Mode (Skip Heavy Initialization)", value=True)

        if test_mode:
            st.info("🧪 Running in test mode - skipping heavy initialization")
            interface = FinancialQAInterface(test_mode=True)
        else:
            interface = FinancialQAInterface(test_mode=False)

        interface.run()
    except Exception as e:
        st.error(f"❌ Critical error running interface: {e}")
        logger.error(f"Interface error: {e}")
        st.error("Please check the console logs for more details.")

if __name__ == "__main__":
    main()
