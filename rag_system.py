# ------------------------------------------
# RAG System Implementation for Financial QA
# Features:
#   1. hybrid retrieval
#   2. memory-augmented retrieval
#   3. response generation
# ------------------------------------------

# -------------------
# Importing libraries
# -------------------
import re
import time
import json
import torch
import faiss
import logging
import chromadb
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryAugmentedRetrieval:
    """Memory-augmented retrieval system for frequently asked questions"""

    def __init__(self, memory_file: str = "memory_bank.json"):
        self.memory_file = Path(memory_file)
        self.memory_bank = self.load_memory_bank()

    def load_memory_bank(self) -> Dict[str, Dict]:
        """Load existing memory bank or create new one"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except:
                logger.warning("Failed to load memory bank, creating new one")

        # --------------------------------------------------
        # Initialize with some common financial Q&A patterns
        # --------------------------------------------------
        return {
            "revenue_questions": {
                "patterns": ["revenue", "sales", "income"],
                "responses": [],
                "frequency": 0
            },
            "profit_questions": {
                "patterns": ["profit", "earnings", "net income"],
                "responses": [],
                "frequency": 0
            },
            "assets_questions": {
                "patterns": ["assets", "balance sheet", "financial position"],
                "responses": [],
                "frequency": 0
            }
        }

    def add_to_memory(self, question: str, answer: str, confidence: float):
        """Add Q&A pair to memory bank"""

        # ----------------------
        # Find matching category
        # ----------------------
        for category, data in self.memory_bank.items():
            if any(pattern in question.lower() for pattern in data["patterns"]):
                data["responses"].append({
                    "question": question,
                    "answer": answer,
                    "confidence": confidence,
                    "timestamp": time.time()
                })
                data["frequency"] += 1
                break
        self.save_memory_bank()

    def retrieve_from_memory(self, question: str) -> Optional[Dict]:
        """Retrieve relevant response from memory bank"""
        for category, data in self.memory_bank.items():
            if any(pattern in question.lower() for pattern in data["patterns"]):
                if data["responses"]:

                    # -----------------------------------------------
                    # Return the most recent high-confidence response
                    # -----------------------------------------------
                    recent_responses = sorted(
                        data["responses"],
                        key=lambda x: (x["confidence"], x["timestamp"]),
                        reverse=True
                    )
                    return recent_responses[0]
        return None

    def save_memory_bank(self):
        """Save memory bank to file"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory_bank, f, indent=2)

class HybridRetriever:
    """Hybrid retrieval system combining dense and sparse methods"""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunks = []
        self.chunk_embeddings = None
        self.faiss_index = None
        self.bm25_index = None

    def add_chunks(self, chunks: List[Dict[str, str]]):
        """Add text chunks to the retriever"""
        self.chunks = chunks

        # ---------------------
        # Prepare text for BM25
        # ---------------------
        texts = [chunk['text'] for chunk in chunks]
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25_index = BM25Okapi(tokenized_texts)

        # ----------------------------
        # Prepare embeddings for FAISS
        # ----------------------------
        logger.info("Generating embeddings for chunks...")
        self.chunk_embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # -----------------
        # Build FAISS index
        # -----------------
        dimension = self.chunk_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.faiss_index.add(self.chunk_embeddings.astype('float32'))

        logger.info(f"Built FAISS index with {len(chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 5, dense_weight: float = 0.7) -> List[Dict]:
        """Hybrid retrieval combining dense and sparse methods"""

        # ----------------
        # Preprocess query
        # ----------------
        query_lower = query.lower()

        # -----------------------------------
        # Dense retrieval (vector similarity)
        # -----------------------------------
        query_embedding = self.embedding_model.encode([query])
        dense_scores, dense_indices = self.faiss_index.search(
            query_embedding.astype('float32'), top_k
        )

        # -----------------------
        # Sparse retrieval (BM25)
        # -----------------------
        tokenized_query = query_lower.split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_scores = bm25_scores[bm25_indices]

        # ------------------------------------
        # Combine results with weighted fusion
        # ------------------------------------
        combined_results = {}

        # -----------------
        # Add dense results
        # -----------------
        for i, (score, idx) in enumerate(zip(dense_scores[0], dense_indices[0])):
            if idx not in combined_results:
                combined_results[idx] = {
                    'chunk': self.chunks[idx],
                    'dense_score': float(score),
                    'bm25_score': 0.0,
                    'combined_score': 0.0
                }
            else:
                combined_results[idx]['dense_score'] = float(score)

        # -----------------
        # Add BM25 results
        # -----------------
        for i, (score, idx) in enumerate(zip(bm25_scores, bm25_indices)):
            if idx not in combined_results:
                combined_results[idx] = {
                    'chunk': self.chunks[idx],
                    'dense_score': 0.0,
                    'bm25_score': float(score),
                    'combined_score': 0.0
                }
            else:
                combined_results[idx]['bm25_score'] = float(score)

        # -------------------------
        # Calculate combined scores
        # -------------------------
        for result in combined_results.values():
            result['combined_score'] = (
                dense_weight * result['dense_score'] +
                (1 - dense_weight) * result['bm25_score']
            )

        # ---------------------------------------------
        # Sort by combined score and return top results
        # ---------------------------------------------
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )

        return sorted_results[:top_k]

class RAGSystem:
    """Complete RAG system with retrieval and generation"""

    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 generation_model: str = "distilgpt2",
                 memory_file: str = "memory_bank.json"):

        self.retriever = HybridRetriever(embedding_model)
        self.memory_system = MemoryAugmentedRetrieval(memory_file)

        # ---------------------------
        # Initialize generation model
        # ---------------------------
        logger.info(f"Loading generation model: {generation_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(generation_model)
        self.generation_model = AutoModelForCausalLM.from_pretrained(generation_model)

        # ----------------------------
        # Set pad token if not present
        # ----------------------------
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generation_model.to(self.device)

        logger.info(f"RAG system initialized on {self.device}")

    def add_documents(self, chunks: List[Dict[str, str]]):
        """Add document chunks to the retriever"""
        self.retriever.add_chunks(chunks)

    def generate_response(self, query: str, retrieved_chunks: List[Dict], max_length: int = 200) -> str:
        """Generate response using retrieved chunks and query"""

        # -------------------------------------
        # Prepare context from retrieved chunks
        # -------------------------------------
        context = " ".join([chunk['chunk']['text'] for chunk in retrieved_chunks[:3]])

        # -------------
        # Create prompt
        # -------------
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

        # --------
        # Tokenize
        # --------
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(self.device)

        # --------
        # Generate
        # --------
        with torch.no_grad():
            outputs = self.generation_model.generate(
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

        # Extract only the answer part
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response[len(prompt):].strip()

        return answer

    def answer_question(self, query: str, top_k: int = 5) -> Dict:
        """Main method to answer a question using RAG"""
        start_time = time.time()

        # ------------------
        # Check memory first
        # ------------------
        memory_response = self.memory_system.retrieve_from_memory(query)
        if memory_response and memory_response['confidence'] > 0.8:
            return {
                'answer': memory_response['answer'],
                'confidence': memory_response['confidence'],
                'method': 'memory',
                'response_time': time.time() - start_time,
                'sources': ['memory_bank']
            }

        # ------------------------
        # Retrieve relevant chunks
        # ------------------------
        retrieved_chunks = self.retriever.retrieve(query, top_k)

        if not retrieved_chunks:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'confidence': 0.0,
                'method': 'rag',
                'response_time': time.time() - start_time,
                'sources': []
            }

        # Generate response
        answer = self.generate_response(query, retrieved_chunks)

        # Calculate confidence based on retrieval scores
        avg_confidence = np.mean([chunk['combined_score'] for chunk in retrieved_chunks])

        # Add to memory for future use
        self.memory_system.add_to_memory(query, answer, avg_confidence)

        response_time = time.time() - start_time

        return {
            'answer': answer,
            'confidence': float(avg_confidence),
            'method': 'rag',
            'response_time': response_time,
            'sources': [chunk['chunk']['source'] for chunk in retrieved_chunks],
            'retrieved_chunks': retrieved_chunks
        }

class InputGuardrail:
    """Input-side guardrail to filter irrelevant or harmful queries"""

    def __init__(self):
        self.financial_keywords = {
            'revenue', 'income', 'profit', 'assets', 'liabilities', 'equity',
            'cash flow', 'balance sheet', 'income statement', 'earnings',
            'financial', 'quarter', 'year', 'fiscal', 'consolidated'
        }

        self.harmful_patterns = [
            r'delete.*file',
            r'format.*disk',
            r'rm.*-rf',
            r'drop.*table',
            r'exec.*system'
        ]

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate if query is relevant and safe"""
        query_lower = query.lower()

        # --------------------------
        # Check for harmful patterns
        # --------------------------
        for pattern in self.harmful_patterns:
            if re.search(pattern, query_lower):
                return False, "Query contains potentially harmful content"

        # -----------------------------------
        # Check if query is financial-related
        # -----------------------------------
        if any(keyword in query_lower for keyword in self.financial_keywords):
            return True, "Query is relevant to financial data"

        # ------------------------------------------------------
        # Check if query is a general question about the company
        # ------------------------------------------------------
        company_keywords = {'apple', 'company', 'business', 'what', 'how', 'when', 'where'}
        if any(keyword in query_lower for keyword in company_keywords):
            return True, "Query is relevant to company information"

        return False, "Query is not relevant to financial or company data"

class OutputGuardrail:
    """Output-side guardrail to detect hallucinated or non-factual outputs"""

    def __init__(self):
        self.factuality_indicators = [
            'i don\'t know',
            'i cannot answer',
            'no information available',
            'data not provided',
            'unclear',
            'unknown'
        ]

    def validate_response(self, response: str, confidence: float) -> Tuple[bool, str]:
        """Validate if response is factual and reliable"""
        response_lower = response.lower()

        # --------------------------
        # Check confidence threshold
        # --------------------------
        if confidence < 0.3:
            return False, "Low confidence response - may be unreliable"

        # --------------------------------
        # Check for uncertainty indicators
        # --------------------------------
        if any(indicator in response_lower for indicator in self.factuality_indicators):
            return False, "Response indicates lack of factual information"

        # ----------------------------------
        # Check for contradictory statements
        # ----------------------------------
        if 'but' in response_lower and 'however' in response_lower:
            return False, "Response contains contradictory statements"

        return True, "Response appears factual and reliable"

if __name__ == "__main__":
    # Test the RAG system
    from data_processor import FinancialDataProcessor

    # Process documents
    processor = FinancialDataProcessor()
    processed_texts, qa_pairs = processor.process_all_documents()
    chunks = processor.get_text_chunks()

    # Initialize RAG system
    rag_system = RAGSystem()
    rag_system.add_documents(chunks)

    # Test questions
    test_questions = [
        "What was the company's revenue in 2024?",
        "What are the total assets?",
        "What is the capital of France?"  # Irrelevant question
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        response = rag_system.answer_question(question)
        print(f"Answer: {response['answer']}")
        print(f"Confidence: {response['confidence']:.3f}")
        print(f"Method: {response['method']}")
        print(f"Response Time: {response['response_time']:.3f}s")
