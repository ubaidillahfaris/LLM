# 🚀 Google Colab Setup Guide

Quick guide untuk run Laravel RAG LLM di Google Colab.

## Method 1: Direct Link (EASIEST)

**Just click this link:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ubaidillahfaris/LLM/blob/main/notebooks/Colab_Quick_Start.ipynb)

Then **run all cells**! Done! 🎉

---

## Method 2: Manual Upload

### Step 1: Upload Repository

```python
# Di Colab, run cell ini:
!git clone https://github.com/ubaidillahfaris/LLM.git /content/LLM
%cd /content/LLM
```

### Step 2: Install Dependencies

```python
!pip install -q transformers torch datasets pandas numpy tqdm
```

### Step 3: Setup Path & Import

```python
import sys
import os

# Setup paths for Colab
project_root = '/content/LLM'
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import modules
from config_loader import ConfigLoader
from retrieval import RAGRetriever
from model_utils import ModelManager, RAGGenerator

print("✅ Imports successful!")
```

### Step 4: Run RAG System

```python
import torch

# Load config
config = ConfigLoader('/content/LLM/configs/config.json')

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_manager = ModelManager(model_name="gpt2", device=device)
model_manager.load_model()

# Setup RAG
retriever = RAGRetriever('/content/LLM/data/knowledge_base/local_db.json')
rag_generator = RAGGenerator(model_manager, retriever)

print("✅ RAG ready!")
```

### Step 5: Ask Questions!

```python
def ask(question):
    result = rag_generator.generate_with_context(
        query=question,
        max_new_tokens=200,
        temperature=0.7
    )
    print(f"Q: {question}")
    print(f"A: {result['answer']}\n")
    return result

# Try it
ask("Bagaimana cara install Laravel?")
ask("Apa itu Eloquent ORM?")
```

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'config_loader'`

**Fix:**

```python
import sys
sys.path.insert(0, '/content/LLM/src')
```

### Error: `No module named 'torch'` or `No module named 'transformers'`

**Fix:**

```python
!pip install transformers torch datasets pandas numpy tqdm
```

Then **restart runtime**: Runtime → Restart runtime

### Error: `FileNotFoundError: config.json not found`

**Fix:**

```python
# Check if repo exists
!ls -la /content/LLM

# If not, clone it:
!git clone https://github.com/ubaidillahfaris/LLM.git /content/LLM
```

### Files di `/content/LLM` tapi import gagal

**Fix:**

```python
# Verify structure
!ls -la /content/LLM/src/
!ls -la /content/LLM/configs/

# Add to path
import sys
sys.path.insert(0, '/content/LLM/src')

# Try import again
from config_loader import ConfigLoader
```

---

## Path Reference untuk Colab

Kalau lu di **Google Colab**, pakai paths ini:

```python
PROJECT_ROOT = '/content/LLM'
CONFIG_PATH = '/content/LLM/configs/config.json'
KB_PATH = '/content/LLM/data/knowledge_base/local_db.json'
DATASET_PATH = '/content/LLM/data/raw/laravel_qa_dataset.json'
SRC_PATH = '/content/LLM/src'
```

Kalau di **local machine**, pakai paths ini:

```python
PROJECT_ROOT = '/home/user/LLM'  # or './LLM'
CONFIG_PATH = './configs/config.json'
KB_PATH = './data/knowledge_base/local_db.json'
DATASET_PATH = './data/raw/laravel_qa_dataset.json'
SRC_PATH = './src'
```

---

## GPU di Colab

Enable GPU untuk faster inference:

1. Runtime → Change runtime type
2. Hardware accelerator → **GPU (T4)**
3. Save
4. Run notebook

Check GPU:

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

---

## Quick Diagnostic

Run this untuk check setup:

```python
import os
import sys

print("🔍 Diagnostic Check")
print("=" * 50)

# Check paths
print(f"Current dir: {os.getcwd()}")
print(f"Project exists: {os.path.exists('/content/LLM')}")
print(f"Config exists: {os.path.exists('/content/LLM/configs/config.json')}")
print(f"Src exists: {os.path.exists('/content/LLM/src')}")

# Check imports
sys.path.insert(0, '/content/LLM/src')
try:
    from config_loader import ConfigLoader
    print("✓ config_loader OK")
except Exception as e:
    print(f"✗ config_loader FAILED: {e}")

try:
    import torch
    print(f"✓ torch OK (GPU: {torch.cuda.is_available()})")
except Exception as e:
    print(f"✗ torch FAILED: {e}")

try:
    from transformers import GPT2LMHeadModel
    print("✓ transformers OK")
except Exception as e:
    print(f"✗ transformers FAILED: {e}")

print("=" * 50)
```

---

## Need Help?

- 📖 [Main README](README.md)
- 🐛 [Report Issues](https://github.com/ubaidillahfaris/LLM/issues)
- 💬 Check error messages carefully - they usually tell you what's wrong!

---

**Happy Coding! 🎉**
