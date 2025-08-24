# 🚀 Quick Start Guide

Get the Financial QA System up and running in minutes!

## ⚡ Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the System
```bash
python test_system.py
```

### 3. Start the Web Interface
```bash
python main.py interface
```

### 4. Open your browser and start asking questions!

## 🔧 What Each Command Does

| Command | Purpose | Time |
|---------|---------|------|
| `python test_system.py` | Verify all components work | 2-3 min |
| `python main.py interface` | Launch web app | 30 sec |
| `python main.py data` | Process documents only | 1-2 min |
| `python main.py rag` | Test RAG system | 2-3 min |
| `python main.py fine-tune` | Test fine-tuning | 5-10 min |
| `python main.py evaluate` | Run full evaluation | 10-15 min |
| `python main.py all` | Complete pipeline | 15-20 min |

## 🎯 First Questions to Try

1. **"What type of company is this?"**
2. **"What was the company's revenue in 2024?"**
3. **"What are the total assets?"**
4. **"What are the main business segments?"**

## 🆘 Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
# Make sure you're in the project root directory
cd financial-qa-system
# Install requirements again
pip install -r requirements.txt
```

**CUDA/GPU issues:**
```bash
# The system works on CPU, just slower
# For GPU acceleration, install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Memory issues:**
```bash
# Reduce batch size in fine-tuning
# Edit src/fine_tune_system.py, change batch_size to 1
```

**PDF processing errors:**
```bash
# Install system dependencies
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## 📊 Expected Results

After running the test, you should see:
- ✅ Documents processed: 3
- ✅ Q&A pairs generated: 10-15
- ✅ Text chunks created: 20-30
- ✅ All systems ready

## 🎉 Success!

If you see all green checkmarks, you're ready to:
1. **Ask questions** in the web interface
2. **Compare RAG vs Fine-tuning** performance
3. **Run evaluations** to see detailed metrics
4. **Explore the code** to understand how it works

## 🔍 Next Steps

- **Read the full README.md** for detailed documentation
- **Try different question types** to see system behavior
- **Run comprehensive evaluation** to see performance metrics
- **Modify parameters** to experiment with different settings

## 💡 Pro Tips

- **Start with simple questions** to verify the system works
- **Use the web interface** for interactive exploration
- **Check the logs** if something goes wrong
- **GPU acceleration** makes fine-tuning much faster
- **Smaller models** work faster but may be less accurate