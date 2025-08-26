<div align="center">

![](https://raw.githubusercontent.com/niranjanjoshi/MLOPS-Pipeline/refs/heads/main/images/BITS_banner.png)

#  S2-24_AIMLCZG521 - Conversational AI | BITS Pilani WILP

## Group No. - 110

| Name | StudentID | Contribution % |
|------|-----------|----------------|
| JOSHI NIRANJAN SURYAKANT  | 2023AC05011 | 100% |
| PRATEEK RALHAN | 2023AC05673 | 100% |
| KESHARKAR SURAJ SANJAY | 2023AD05004 | 100% |
| SAURABH SUNIT JOTSHI | 2023AC05565 | 100%  |
| KILLI SATYA PRAKASH | 2023AC05066 | 100%  |

# 📊 Financial QA System: RAG vs Fine-tuning Comparison [![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

</div>

*A comprehensive comparative analysis system that implements and evaluates two approaches for answering questions based on company financial statements:*

1. **Retrieval-Augmented Generation (RAG) Chatbot**: Combines document retrieval and generative response
2. **Fine-Tuned Language Model (FT) Chatbot**: Directly fine-tunes a small open-source language model on financial Q&A

👉 [🎬 Live WebApp🔗](https://huggingface.co/spaces/vibertron/Financial_QnA)

👉 [📝 Architecture Summary Document](./docs/summary.pdf)

## 🎯 Objective

Develop and compare two systems for answering questions based on company financial statements (last two years) using the same financial data for both methods and perform a detailed comparison on accuracy, speed, and robustness.

## ✨ Key Features

### 🔍 RAG System
- **Hybrid Retrieval**: Combines dense (vector) and sparse (BM25) retrieval methods
- **Memory-Augmented Retrieval**: Persistent memory bank for frequently asked questions
- **Advanced Guardrails**: Input and output validation systems
- **Multi-source Retrieval**: FAISS vector database + ChromaDB integration
- **Document Chunking**: Intelligent text segmentation with configurable chunk sizes

### 🎯 Fine-Tuned System
- **Continual Learning**: Incremental fine-tuning without catastrophic forgetting
- **Domain Adaptation**: Specialized for financial Q&A domain
- **Efficient Training**: Optimized hyperparameters for small models
- **Confidence Scoring**: Built-in confidence estimation
- **Model Persistence**: Save and load fine-tuned models

### 📊 Evaluation & Comparison
- **Comprehensive Metrics**: Accuracy, response time, confidence, factuality
- **Visualization**: Interactive charts and performance comparisons
- **Test Suite**: Diverse question types (relevant high/low confidence, irrelevant)
- **ROUGE Scoring**: Text similarity metrics for quality assessment

### 🖥️ User Interface
- **Streamlit Web App**: Modern, responsive interface
- **Real-time Comparison**: Side-by-side RAG vs Fine-tuned results
- **Interactive QA**: Ask questions and get instant responses
- **Performance Dashboard**: Live metrics and visualizations

## 🏗️ System Architecture

```
Financial QA System
├── Data Processing
│   ├── PDF Extraction (pdfplumber, PyPDF2)
│   ├── Text Cleaning & Segmentation
│   ├── Q&A Pair Generation
│   └── Chunking for RAG
├── RAG System
│   ├── Hybrid Retrieval (FAISS + BM25)
│   ├── Memory-Augmented Retrieval
│   ├── Response Generation (DistilGPT2)
│   └── Guardrails (Input/Output)
├── Fine-Tuned System
│   ├── Continual Learning
│   ├── Domain Adaptation
│   ├── Model Training & Persistence
│   └── Confidence Estimation
├── Evaluation System
│   ├── Performance Metrics
│   ├── Comparative Analysis
│   ├── Visualization Generation
│   └── Results Export
└── User Interface
    ├── Streamlit Web App
    ├── Interactive QA
    ├── System Comparison
    └── Performance Dashboard
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd financial-qa-system
```

### 2. Create Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required Models
The system will automatically download required models on first run:
- `all-MiniLM-L6-v2` (sentence embeddings)
- `distilgpt2` (generation model)
- `distilbert-base-uncased` (classification)

## 📖 Usage

### Command Line Interface

#### 1. Data Processing Only
```bash
python main.py data
```

#### 2. RAG System Only
```bash
python main.py rag
```

#### 3. Fine-tuning Only
```bash
python main.py fine-tune
```

#### 4. Comprehensive Evaluation
```bash
python main.py evaluate
```

#### 5. Web Interface
```bash
python main.py interface
```

#### 6. Complete Pipeline
```bash
python main.py all
```

### Web Interface

1. **Start the interface**:
   ```bash
   python main.py interface
   ```

2. **Open your browser** and navigate to the displayed URL

3. **Select your preferred system**:
   - RAG System
   - Fine-tuned System
   - Both (Comparison)

4. **Ask questions** and view results in real-time

## 📊 System Comparison

### RAG System Strengths
- **Adaptability**: Easy to update with new documents
- **Factual Grounding**: Direct access to source documents
- **Transparency**: Clear source attribution
- **Flexibility**: Handles diverse question types

### Fine-Tuned System Strengths
- **Speed**: Faster inference after training
- **Fluency**: More natural, coherent responses
- **Efficiency**: Lower computational overhead
- **Specialization**: Domain-specific knowledge

### Trade-offs
- **RAG**: Higher accuracy, slower response, more resource-intensive
- **Fine-tuned**: Lower accuracy, faster response, more efficient

## 🔧 Configuration

### Training Parameters
```python
@dataclass
class TrainingConfig:
    model_name: str = "distilgpt2"
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 512
    warmup_steps: int = 100
    weight_decay: float = 0.01
```

### RAG Parameters
- **Chunk Size**: Configurable text segmentation (100-400 tokens)
- **Top-K Retrieval**: Number of chunks to retrieve (default: 5)
- **Dense Weight**: Weight for vector similarity vs BM25 (default: 0.7)

## 📈 Evaluation Metrics

### Performance Metrics
- **Accuracy**: Correct answer rate
- **Response Time**: Average inference speed
- **Confidence**: Model confidence scores
- **Factuality**: Response reliability assessment

### Quality Metrics
- **ROUGE Scores**: Text similarity metrics
- **Source Attribution**: Document source tracking
- **Validation Status**: Input/output guardrail results

## 📁 Project Structure

```
financial-qa-system/
├── src/
│   ├── __init__.py
│   ├── data_processor.py      # Document processing & Q&A generation
│   ├── rag_system.py          # RAG implementation
│   ├── fine_tune_system.py    # Fine-tuning implementation
│   ├── evaluation_system.py   # Evaluation & comparison
│   └── interface.py           # Streamlit web interface
├── financial_statements/      # Input PDF documents
├── processed_data/            # Processed texts & Q&A pairs
├── evaluation_results/        # Evaluation outputs & visualizations
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🧪 Testing

### Test Questions Categories
1. **Relevant, High-Confidence**: Clear facts in financial data
2. **Relevant, Low-Confidence**: Ambiguous or sparse information
3. **Irrelevant**: Questions outside financial scope

### Example Test Questions
- "What was the company's revenue in 2024?"
- "What are the total assets?"
- "What type of company is this?"
- "What is the capital of France?" (irrelevant)

## 🔒 Guardrails

### Input Guardrails
- **Relevance Check**: Validates financial/company-related queries
- **Harmful Content**: Filters potentially dangerous inputs
- **Query Validation**: Ensures proper question format

### Output Guardrails
- **Factuality Check**: Detects hallucinated responses
- **Confidence Threshold**: Flags low-confidence outputs
- **Contradiction Detection**: Identifies conflicting statements

## 🚀 Advanced Features

### Memory-Augmented Retrieval
- Persistent memory bank for frequent Q&A patterns
- Automatic categorization and retrieval
- Confidence-based response selection

### Continual Learning
- Incremental fine-tuning on new data
- Catastrophic forgetting prevention
- Domain adaptation capabilities

### Hybrid Retrieval
- Dense retrieval (sentence embeddings)
- Sparse retrieval (BM25)
- Weighted score fusion

## 📊 Results Example

| Question              | Method    | Answer            | Confidence | Time (s) | Correct (Y/N) |
|-----------------------|-----------|-------------------|------------|----------|---------------|
| Revenue in 2024?      | RAG       | $391.0B           | 0.93       | 9.11     | Y             |
| Revenue in 2024?      | Fine-Tune | $391.0B           | 0.91       | 21.23    | Y             |
| Total sales(iphones)? | RAG       | $182.2B           | 0.89       | 4.22     | N             |
| Total sales(iphones)? | Fine-Tune | $201.2B           | 0.92       | 44.12    | Y             |
| Capital of France?    | RAG       | blank response    | 0.35       | 11.2     | Y             |
| Capital of France?    | Fine-Tune | Paris             | 0.22       | 3.47     | N             |

## 🙏 Acknowledgments

- **Hugging Face**: Transformers library and model hub
- **Sentence Transformers**: Embedding models
- **FAISS**: Vector similarity search
- **Streamlit**: Web interface framework
- **Apple Inc.**: Financial statement data for testing

## 🔮 Future Enhancements

- **Multi-modal Support**: Image and table extraction from PDFs
- **Real-time Updates**: Live document ingestion and processing
- **Advanced Guardrails**: More sophisticated validation systems
- **Model Compression**: Quantization and distillation for efficiency
- **API Integration**: RESTful API for external applications

- **Multi-language Support**: Internationalization capabilities

