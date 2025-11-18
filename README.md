# Laravel RAG LLM 🚀

RAG (Retrieval Augmented Generation) system untuk Laravel development menggunakan GPT-2. Project ini membantu developer untuk bertanya tentang Laravel dan mendapatkan jawaban yang akurat dengan kombinasi knowledge base dan AI generation.

## 🌟 Features

- **RAG System**: Kombinasi retrieval dari knowledge base + GPT-2 generation
- **Laravel-Specific**: Fine-tuned untuk Laravel development questions
- **Modular Architecture**: Clean separation of concerns (data processing, retrieval, model utils)
- **Easy to Extend**: Mudah menambah dataset dan knowledge baru
- **Interactive Notebook**: Jupyter notebook untuk experimentation
- **Configurable**: JSON-based configuration system

## 📁 Project Structure

```
LLM/
├── data/
│   ├── raw/                          # Raw Laravel QA dataset
│   │   └── laravel_qa_dataset.json
│   ├── processed/                    # Processed training data
│   │   └── training_data.json
│   └── knowledge_base/               # Local knowledge base
│       └── local_db.json
├── models/
│   └── fine_tuned_gpt2/             # Fine-tuned model akan disimpan di sini
├── src/
│   ├── __init__.py
│   ├── config_loader.py             # Configuration loader
│   ├── data_processing.py           # Data processing utilities
│   ├── retrieval.py                 # RAG retrieval system
│   └── model_utils.py               # Model loading & generation
├── notebooks/
│   └── Laravel_RAG_LLM_Complete.ipynb  # Main notebook
├── configs/
│   └── config.json                  # Configuration file
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/ubaidillahfaris/LLM.git
cd LLM

# Install dependencies
pip install -r requirements.txt
```

### 2. Open Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/Laravel_RAG_LLM_Complete.ipynb
```

### 3. Run Cells

Ikuti step-by-step di notebook:
1. Setup & Installation
2. Load Configuration
3. Explore Dataset
4. Process Data
5. Load Model
6. (Optional) Train Model
7. Setup RAG System
8. Test Inference
9. Interactive Demo

## 📊 Dataset

Dataset Laravel QA sudah include 15+ pertanyaan umum tentang:
- Installation & Setup
- Eloquent ORM
- Controllers & Routes
- Migrations & Seeding
- Middleware & Authentication
- Blade Templates
- Validation
- API Development
- Queue & Caching

### Menambah Dataset

Edit file `data/raw/laravel_qa_dataset.json`:

```json
{
  "id": 16,
  "question": "Pertanyaan baru?",
  "answer": "Jawaban lengkap...",
  "category": "kategori",
  "difficulty": "beginner|intermediate|advanced"
}
```

Kemudian jalankan data processing di notebook.

## ⚙️ Configuration

Edit `configs/config.json` untuk customize:

```json
{
  "model": {
    "name": "gpt2",
    "temperature": 0.7,
    "max_length": 512
  },
  "training": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "learning_rate": 5e-5
  },
  "generation": {
    "max_new_tokens": 200,
    "temperature": 0.7
  }
}
```

## 🎯 Usage Examples

### Basic Usage

```python
from src.model_utils import ModelManager, RAGGenerator
from src.retrieval import RAGRetriever

# Load model
model_manager = ModelManager(model_name="gpt2")
model_manager.load_model()

# Setup RAG
retriever = RAGRetriever(kb_path="./data/knowledge_base/local_db.json")
rag_generator = RAGGenerator(model_manager, retriever)

# Ask question
result = rag_generator.generate_with_context(
    query="Bagaimana cara membuat controller di Laravel?",
    max_new_tokens=200
)

print(result['answer'])
```

### Add New Knowledge

```python
from src.retrieval import KnowledgeBase

kb = KnowledgeBase()
kb.add_entry(
    "cara deploy laravel",
    "Untuk deploy Laravel: 1) Setup server dengan PHP 8.1+..."
)
```

## 🔧 Training

Untuk fine-tune model dengan dataset Anda:

1. Tambah data ke `data/raw/laravel_qa_dataset.json`
2. Process data dengan `DataProcessor`
3. Uncomment training cell di notebook
4. Run training (akan memakan waktu 10-30 menit)

Model akan di-save ke `models/fine_tuned_gpt2/`

## 📈 Performance Tips

1. **GPU**: Gunakan GPU untuk training lebih cepat
2. **Batch Size**: Sesuaikan batch size dengan VRAM Anda
3. **Dataset**: Lebih banyak data = model lebih baik
4. **Temperature**: Lower temperature (0.3-0.5) = more focused, higher (0.7-0.9) = more creative

## 🛠️ Advanced Features

### Semantic Search (Coming Soon)

```python
# TODO: Implement dengan sentence-transformers
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')
# ... implement semantic search
```

### Web API (Coming Soon)

```python
# TODO: FastAPI endpoint
from fastapi import FastAPI

app = FastAPI()

@app.post("/ask")
def ask_question(query: str):
    result = rag_generator.generate_with_context(query)
    return result
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch
3. Add tests
4. Submit pull request

## 📝 TODO

- [ ] Add more Laravel QA pairs (target: 100+)
- [ ] Implement semantic search dengan embeddings
- [ ] Add web interface (FastAPI + React)
- [ ] Add unit tests
- [ ] Docker containerization
- [ ] Deploy to cloud

## 📚 Resources

- [Laravel Documentation](https://laravel.com/docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [RAG Paper](https://arxiv.org/abs/2005.11401)

## 📄 License

MIT License

## 👨‍💻 Author

**Ubaidillah Faris**
- GitHub: [@ubaidillahfaris](https://github.com/ubaidillahfaris)

---

**Happy Coding! 🎉**