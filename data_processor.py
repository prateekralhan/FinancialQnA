# -------------------------------------------------------------
# Data Processing Module for Financial QA System
# Handles PDF extraction, text cleaning, and Q&A pair generation
# -------------------------------------------------------------

# -------------------
# Importing libraries
# -------------------
import os
import re
import json
import nltk
import PyPDF2
import logging
import pdfplumber
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from typing import List, Dict, Tuple, Optional
from nltk.tokenize import sent_tokenize, word_tokenize
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Download required NLTK data
# ---------------------------
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

class FinancialDataProcessor:
    """Processes financial documents and generates Q&A pairs"""

    def __init__(self, data_dir: str = "financial_statements"):
        self.data_dir = Path(data_dir)
        self.processed_texts = {}
        self.qa_pairs = []
        self.stop_words = set(stopwords.words('english'))

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""

        # ------------------------------------------------------
        # Try pdfplumber first (better for structured documents)
        # ------------------------------------------------------
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            logger.info(f"Successfully extracted text using pdfplumber from {pdf_path.name}")
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path.name}: {e}")

            # ------------------
            # Fallback to PyPDF2
            # ------------------
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                logger.info(f"Successfully extracted text using PyPDF2 from {pdf_path.name}")
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed for {pdf_path.name}: {e2}")
                return ""

        return text

    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing noise and formatting"""

        # -------------------------------------
        # Remove extra whitespace and normalize
        # -------------------------------------
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\$\-\(\)\%]', '', text)

        # -------------------------------
        # Remove page numbers and headers
        # -------------------------------
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)

        # ------------------------------------------
        # Remove common financial document artifacts
        # ------------------------------------------
        text = re.sub(r'CONSOLIDATED|FINANCIAL STATEMENTS|QUARTER ENDED|YEAR ENDED', '', text, flags=re.IGNORECASE)

        return text.strip()

    def segment_financial_sections(self, text: str) -> Dict[str, str]:
        """Segment text into logical financial sections"""
        sections = {
            'income_statement': '',
            'balance_sheet': '',
            'cash_flow': '',
            'notes': ''
        }

        # ---------------------------------
        # Simple keyword-based segmentation
        # ---------------------------------
        lines = text.split('\n')
        current_section = 'notes'

        for line in lines:
            line_lower = line.lower()

            if any(keyword in line_lower for keyword in ['revenue', 'income', 'earnings', 'net income']):
                current_section = 'income_statement'
            elif any(keyword in line_lower for keyword in ['assets', 'liabilities', 'equity', 'total assets']):
                current_section = 'balance_sheet'
            elif any(keyword in line_lower for keyword in ['cash flow', 'operating activities', 'investing activities']):
                current_section = 'cash_flow'

            sections[current_section] += line + '\n'

        return sections

    def extract_financial_metrics(self, text: str) -> Dict[str, str]:
        """Extract key financial metrics from text"""
        metrics = {}

        # ----------------
        # Revenue patterns
        # ----------------
        revenue_patterns = [
            r'revenue.*?(\$[\d,]+\.?\d*)\s*(billion|million|thousand)?',
            r'total revenue.*?(\$[\d,]+\.?\d*)\s*(billion|million|thousand)?',
            r'net revenue.*?(\$[\d,]+\.?\d*)\s*(billion|million|thousand)?'
        ]

        for pattern in revenue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics['revenue'] = matches[0][0] + ' ' + (matches[0][1] or '')
                break

        # -------------------
        # Net income patterns
        # -------------------
        net_income_patterns = [
            r'net income.*?(\$[\d,]+\.?\d*)\s*(billion|million|thousand)?',
            r'net earnings.*?(\$[\d,]+\.?\d*)\s*(billion|million|thousand)?'
        ]

        for pattern in net_income_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics['net_income'] = matches[0][0] + ' ' + (matches[0][1] or '')
                break

        # ---------------
        # Assets patterns
        # ---------------
        assets_patterns = [
            r'total assets.*?(\$[\d,]+\.?\d*)\s*(billion|million|thousand)?',
            r'assets.*?(\$[\d,]+\.?\d*)\s*(billion|million|thousand)?'
        ]

        for pattern in assets_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics['total_assets'] = matches[0][0] + ' ' + (matches[0][1] or '')
                break

        return metrics

    def generate_qa_pairs(self, processed_texts: Dict[str, str]) -> List[Dict[str, str]]:
        """Generate Q&A pairs based on extracted financial data"""
        qa_pairs = []

        # ----------------------------------
        # Extract metrics from all documents
        # ----------------------------------
        all_metrics = {}
        for doc_name, text in processed_texts.items():
            metrics = self.extract_financial_metrics(text)
            all_metrics[doc_name] = metrics

        # ------------------------------
        # Generate Q&A pairs for revenue
        # ------------------------------
        for doc_name, metrics in all_metrics.items():
            if 'revenue' in metrics:
                year = doc_name.split('_')[0] if '_' in doc_name else '2024'
                qa_pairs.append({
                    'question': f'What was the company\'s revenue in {year}?',
                    'answer': f'The company\'s revenue in {year} was {metrics["revenue"]}.',
                    'source': doc_name,
                    'category': 'revenue'
                })

        # ---------------------------------
        # Generate Q&A pairs for net income
        # ---------------------------------
        for doc_name, metrics in all_metrics.items():
            if 'net_income' in metrics:
                year = doc_name.split('_')[0] if '_' in doc_name else '2024'
                qa_pairs.append({
                    'question': f'What was the company\'s net income in {year}?',
                    'answer': f'The company\'s net income in {year} was {metrics["net_income"]}.',
                    'source': doc_name,
                    'category': 'net_income'
                })

        # -----------------------------------
        # Generate Q&A pairs for total assets
        # -----------------------------------
        for doc_name, metrics in all_metrics.items():
            if 'total_assets' in metrics:
                year = doc_name.split('_')[0] if '_' in doc_name else '2024'
                qa_pairs.append({
                    'question': f'What were the company\'s total assets in {year}?',
                    'answer': f'The company\'s total assets in {year} were {metrics["total_assets"]}.',
                    'source': doc_name,
                    'category': 'total_assets'
                })

        # ------------------------------------
        # Add some general financial questions
        # ------------------------------------
        qa_pairs.extend([
            {
                'question': 'What type of company is this?',
                'answer': 'This is Apple Inc., a technology company that designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories.',
                'source': 'general',
                'category': 'company_info'
            },
            {
                'question': 'What are the main business segments?',
                'answer': 'Apple\'s main business segments include iPhone, Mac, iPad, Wearables, Home and Accessories, and Services.',
                'source': 'general',
                'category': 'business_segments'
            }
        ])

        return qa_pairs

    def process_all_documents(self) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        """Process all financial documents and generate Q&A pairs"""
        logger.info("Starting document processing...")

        # ---------------------
        # Process each PDF file
        # ---------------------
        for pdf_file in self.data_dir.glob("*.pdf"):
            logger.info(f"Processing {pdf_file.name}...")

            # ------------
            # Extract text
            # ------------
            raw_text = self.extract_text_from_pdf(pdf_file)
            if not raw_text:
                continue

            # ----------
            # Clean text
            # ----------
            cleaned_text = self.clean_text(raw_text)

            # --------------------
            # Store processed text
            # --------------------
            doc_name = pdf_file.stem
            self.processed_texts[doc_name] = cleaned_text

            logger.info(f"Successfully processed {pdf_file.name}")

        # ------------------
        # Generate Q&A pairs
        # ------------------
        dynamic_qa_pairs = self.generate_qa_pairs(self.processed_texts)

        # ----------------------------------
        # Load static Q&A pairs if available
        # ----------------------------------
        static_qa_pairs = []
        static_qa_path = "processed_data/qa_pairs_static.json"
        try:
            with open(static_qa_path, 'r', encoding='utf-8') as f:
                static_qa_pairs = json.load(f)
            logger.info(f"Loaded {len(static_qa_pairs)} static Q&A pairs from {static_qa_path}")
        except Exception as e:
            logger.warning(f"Could not load static Q&A pairs: {e}")

        # --------------------------------
        # Concatenate static + dynamic Q&A
        # --------------------------------
        self.qa_pairs = static_qa_pairs + dynamic_qa_pairs
        logger.info(f"Generated {len(self.qa_pairs)} Q&A pairs")

        return self.processed_texts, self.qa_pairs

    def save_processed_data(self, output_dir: str = "processed_data"):
        """Save processed texts and Q&A pairs"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # --------------------
        # Save processed texts
        # --------------------
        for doc_name, text in self.processed_texts.items():
            with open(output_path / f"{doc_name}_processed.txt", 'w', encoding='utf-8') as f:
                f.write(text)

        # --------------
        # Save Q&A pairs
        # --------------
        with open(output_path / "qa_pairs.json", 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, indent=2)

        # ---------------------------
        # Save as CSV for fine-tuning
        # ---------------------------
        qa_df = pd.DataFrame(self.qa_pairs)
        qa_df.to_csv(output_path / "qa_pairs.csv", index=False)

        logger.info(f"Saved processed data to {output_dir}")

    def get_text_chunks(self, chunk_size: int = 400, overlap: int = 50) -> List[Dict[str, str]]:
        """Split processed texts into chunks for RAG"""
        chunks = []

        for doc_name, text in self.processed_texts.items():
            sentences = sent_tokenize(text)
            current_chunk = ""
            chunk_id = 0

            for sentence in sentences:
                if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk.strip():
                        chunks.append({
                            'id': f"{doc_name}_chunk_{chunk_id}",
                            'text': current_chunk.strip(),
                            'source': doc_name,
                            'chunk_size': len(current_chunk.split())
                        })
                        chunk_id += 1
                        current_chunk = sentence + " "

            # ------------------
            # Add the last chunk
            # ------------------
            if current_chunk.strip():
                chunks.append({
                    'id': f"{doc_name}_chunk_{chunk_id}",
                    'text': current_chunk.strip(),
                    'source': doc_name,
                    'chunk_size': len(current_chunk.split())
                })

        return chunks

if __name__ == "__main__":

    # -----------------------
    # Test the data processor
    # -----------------------
    processor = FinancialDataProcessor()
    processed_texts, qa_pairs = processor.process_all_documents()
    processor.save_processed_data()

    # ---------------
    # Generate chunks
    # ---------------
    chunks = processor.get_text_chunks()
    print(f"Generated {len(chunks)} text chunks")
    print(f"Generated {len(qa_pairs)} Q&A pairs")
