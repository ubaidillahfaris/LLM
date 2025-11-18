# 🤖 Auto Fine-Tuning Guide

Complete guide untuk automatic fine-tuning dengan human-in-the-loop approval system.

## 🎯 Overview

System ini memungkinkan lu untuk:
1. **Scrape** data Laravel dari internet otomatis
2. **Generate** QA pairs dengan AI
3. **Review & Approve** dengan checkbox form yang user-friendly
4. **Auto fine-tune** model dengan approved data
5. **Continuous learning** loop

---

## 🚀 Quick Start

### Option 1: Interactive Notebook (RECOMMENDED)

```bash
# Open Jupyter
jupyter notebook

# Open: notebooks/Interactive_QA_Review.ipynb
# Run all cells step-by-step
```

### Option 2: Python Scripts

```python
from src.web_scraper import LaravelDataScraper
from src.qa_generator import QAGenerator
from src.auto_finetune import AutoFineTuner

# 1. Scrape data
scraper = LaravelDataScraper()
data = scraper.scrape_laravel_docs(['eloquent', 'routing'])

# 2. Generate QA
qa_gen = QAGenerator()
qa_pairs = qa_gen.batch_generate_from_scraped_data(data)

# 3. Approve QA (manual or via interface)
qa_gen.mark_as_approved(qa_id="...")

# 4. Fine-tune
auto_tuner = AutoFineTuner(model_manager)
result = auto_tuner.start_finetuning(qa_gen.get_approved_qa())
```

---

## 📋 Complete Workflow

### Step 1: Scrape Data dari Internet

**Sources yang supported:**
- ✅ Laravel Official Documentation
- ✅ StackOverflow (via API)
- ✅ Medium (via RSS)
- ✅ Laracasts (via API)

```python
from src.web_scraper import LaravelDataScraper

scraper = LaravelDataScraper()

# Scrape Laravel docs
topics = ['eloquent', 'routing', 'middleware', 'validation']
data = scraper.scrape_laravel_docs(topics)

# Save scraped data
scraper.save_scraped_data('./data/raw/scraped_content.json')

print(f"Scraped {len(data)} sections")
```

**Output:**
```json
{
  "source": "laravel_docs",
  "topic": "eloquent",
  "title": "Getting Started",
  "content": "Eloquent ORM adalah...",
  "url": "https://laravel.com/docs/10.x/eloquent"
}
```

---

### Step 2: Generate QA Pairs Otomatis

```python
from src.qa_generator import QAGenerator

qa_gen = QAGenerator()

# Generate dari scraped content
qa_pairs = qa_gen.batch_generate_from_scraped_data(data)

print(f"Generated {len(qa_pairs)} QA pairs")

# Show pending review
pending = qa_gen.get_pending_qa()
for qa in pending[:3]:
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer'][:100]}...\n")
```

**Generated QA Format:**
```json
{
  "id": "gen_12345",
  "question": "Bagaimana cara kerja Eloquent ORM?",
  "answer": "Eloquent ORM adalah Active Record pattern...",
  "source": "laravel_docs",
  "topic": "eloquent",
  "approved": null,
  "confidence": 0.7
}
```

---

### Step 3: Review & Approve (Interactive Form)

**Notebook Interface:**

Buka `Interactive_QA_Review.ipynb` dan run review cell. Lu bakal lihat:

```
┌─────────────────────────────────────────┐
│ Review Progress: 0 / 15                 │
├─────────────────────────────────────────┤
│ ❓ Question:                            │
│ Bagaimana cara install Laravel?         │
│                                          │
│ 💡 Answer:                              │
│ Untuk install Laravel: 1) Install PHP...│
├─────────────────────────────────────────┤
│ Edit if needed:                         │
│ Question: [text area]                   │
│ Answer:   [text area]                   │
├─────────────────────────────────────────┤
│ [✅ Approve] [❌ Reject] [⏭️ Skip]      │
└─────────────────────────────────────────┘
```

**Actions:**
- **✅ Approve**: Add ke training dataset
- **❌ Reject**: Buang QA pair ini
- **⏭️ Skip**: Review nanti
- **Edit**: Edit question/answer sebelum approve

**Programmatic Approval:**

```python
# Approve
qa_gen.mark_as_approved(qa_id="gen_12345")

# Reject
qa_gen.mark_as_rejected(qa_id="gen_12346", reason="Jawaban kurang lengkap")

# Edit then approve
qa_gen.edit_qa(
    qa_id="gen_12347",
    new_question="Pertanyaan yang lebih jelas",
    new_answer="Jawaban yang lebih lengkap"
)
qa_gen.mark_as_approved("gen_12347")
```

---

### Step 4: View Statistics

```python
stats = qa_gen.get_stats()

print(f"Total Generated: {stats['total_generated']}")
print(f"Approved: {stats['approved']} ({stats['approval_rate']:.1f}%)")
print(f"Rejected: {stats['rejected']}")
print(f"Pending: {stats['pending_review']}")
```

**Output:**
```
Total Generated: 50
Approved: 35 (70.0%)
Rejected: 10
Pending: 5
```

---

### Step 5: Save Approved QA

```python
# Save ke training dataset format
training_data = qa_gen.save_to_training_dataset(
    filepath='./data/raw/approved_qa.json',
    approved_only=True
)

print(f"Saved {len(training_data)} approved QA pairs")
```

---

### Step 6: Auto Fine-Tune

```python
from src.auto_finetune import AutoFineTuner
from src.model_utils import ModelManager

# Load model
model_manager = ModelManager(model_name="gpt2")
model_manager.load_model()

# Initialize auto-tuner
auto_tuner = AutoFineTuner(
    model_manager=model_manager,
    base_dataset_path='./data/raw/laravel_qa_dataset.json',
    approved_qa_path='./data/raw/approved_qa.json',
    training_output_dir='./models/auto_finetuned'
)

# Get approved QA
approved_qa = qa_gen.get_approved_qa()

# Start fine-tuning
result = auto_tuner.start_finetuning(
    approved_qa=approved_qa,
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5
)

print(f"Training complete!")
print(f"Model saved to: {result['model_output_dir']}")
```

**Training Output:**
```
📊 Base dataset: 15 QA pairs
📊 Approved new QA: 35 pairs
✅ Added 35 new unique QA pairs
📊 Total training data: 50 pairs

🏋️  Training model...
Epoch 1/3: [====================] 100%
Epoch 2/3: [====================] 100%
Epoch 3/3: [====================] 100%

✅ Training completed in 243.5 seconds
📦 Model saved to: ./models/auto_finetuned
```

---

### Step 7: Test Fine-Tuned Model

```python
# Load fine-tuned model
finetuned_model = ModelManager(
    model_name="gpt2",
    model_path='./models/auto_finetuned'
)
finetuned_model.load_model()

# Test
def ask(question):
    prompt = f"Question: {question}\nAnswer:"
    response = finetuned_model.generate_response(
        prompt=prompt,
        max_new_tokens=200,
        temperature=0.7
    )
    print(f"Q: {question}")
    print(f"A: {response}\n")

# Try it
ask("Bagaimana cara install Laravel?")
ask("Apa itu Eloquent ORM?")
```

---

## 🔄 Continuous Learning Loop

```python
# 1. Scrape new data (weekly/monthly)
scraper = LaravelDataScraper()
new_data = scraper.scrape_laravel_docs(['new_topics'])

# 2. Generate QA
qa_gen = QAGenerator()
qa_gen.batch_generate_from_scraped_data(new_data)

# 3. Review & approve (manual or automated)
# ... use interactive interface ...

# 4. Auto fine-tune if threshold met
approved = qa_gen.get_approved_qa()
if len(approved) >= 10:  # Minimum threshold
    auto_tuner.start_finetuning(approved)
```

---

## ⚙️ Advanced Configuration

### Custom Data Sources

```python
class CustomScraper(LaravelDataScraper):
    def scrape_custom_source(self, url):
        # Your custom scraping logic
        response = requests.get(url)
        # ... extract content ...
        return scraped_data

scraper = CustomScraper()
data = scraper.scrape_custom_source('https://your-source.com')
```

### Custom QA Generation

```python
class CustomQAGenerator(QAGenerator):
    def generate_qa_from_content(self, content):
        # Your custom QA generation logic
        # Maybe use GPT-4 API for better quality
        qa_pairs = []
        # ... generate QA ...
        return qa_pairs

qa_gen = CustomQAGenerator(model_manager=your_model)
```

### Scheduled Training

```python
from apscheduler.schedulers.background import BackgroundScheduler

def scheduled_training():
    # Check for new approved QA
    approved = qa_gen.get_approved_qa()

    if len(approved) >= 20:  # Threshold
        print("Starting scheduled fine-tuning...")
        auto_tuner.start_finetuning(approved)

# Schedule every week
scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_training, 'interval', weeks=1)
scheduler.start()
```

---

## 📊 Monitoring & Analytics

### Training History

```python
stats = auto_tuner.get_training_stats()

print(f"Total trainings: {stats['total_trainings']}")
print(f"Total QA added: {stats['total_qa_added']}")
print(f"Avg training time: {stats['avg_training_time']:.2f}s")
print(f"Latest model: {stats['latest_model']}")
```

### QA Quality Metrics

```python
# Approval rate by source
sources = {}
for qa in qa_gen.generated_qa:
    source = qa.get('source', 'unknown')
    if source not in sources:
        sources[source] = {'total': 0, 'approved': 0}

    sources[source]['total'] += 1
    if qa.get('approved'):
        sources[source]['approved'] += 1

for source, stats in sources.items():
    rate = stats['approved'] / stats['total'] * 100
    print(f"{source}: {rate:.1f}% approval rate")
```

---

## 🔐 Best Practices

### 1. Data Quality

- ✅ Always review generated QA before approval
- ✅ Edit untuk improve quality
- ✅ Reject low-quality atau incorrect answers
- ✅ Maintain consistent format

### 2. Training Strategy

- ✅ Start dengan small batch (10-20 QA) untuk testing
- ✅ Gradually increase dataset size
- ✅ Regular fine-tuning (weekly/monthly)
- ✅ Keep base dataset clean

### 3. Version Control

```bash
# Tag setiap fine-tuned model
git tag -a v1.0-finetune-2024-01 -m "Fine-tuned dengan 50 approved QA"

# Backup model
cp -r models/auto_finetuned models/backups/v1.0
```

### 4. Testing

- ✅ Test fine-tuned model sebelum deploy
- ✅ Compare dengan base model
- ✅ Check untuk overfitting
- ✅ Validate dengan test set

---

## 🐛 Troubleshooting

### Issue: Scraping Blocked

**Error:** `403 Forbidden` or `429 Too Many Requests`

**Fix:**
```python
# Add delays
time.sleep(2)  # Wait 2s between requests

# Use API instead of scraping
# StackExchange API: https://api.stackexchange.com/docs
```

### Issue: Low Quality QA

**Error:** Generated QA pairs are nonsense

**Fix:**
- Use better prompts
- Filter by confidence score
- Manual review sebelum approve
- Use GPT-4 API untuk generation (better quality)

### Issue: Training OOM (Out of Memory)

**Error:** CUDA out of memory

**Fix:**
```python
# Reduce batch size
auto_tuner.start_finetuning(
    approved_qa=approved,
    batch_size=2,  # Reduce from 4
    gradient_accumulation_steps=4  # Increase
)
```

### Issue: Model Performance Degraded

**Error:** Fine-tuned model worse than base

**Fix:**
- Reduce epochs (overfitting)
- Lower learning rate
- Add validation set
- Check data quality

---

## 📚 Resources

- **Web Scraping**: [BeautifulSoup Docs](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- **StackExchange API**: [API Documentation](https://api.stackexchange.com/docs)
- **Fine-Tuning**: [HuggingFace Guide](https://huggingface.co/docs/transformers/training)
- **Jupyter Widgets**: [ipywidgets Docs](https://ipywidgets.readthedocs.io/)

---

## 🎯 Next Steps

1. **Expand Data Sources**: Add GitHub issues, Reddit, Discord
2. **Improve QA Generation**: Use GPT-4 API atau Claude
3. **Build Web UI**: Replace notebook dengan Flask/FastAPI app
4. **Deploy**: Serve fine-tuned model as API
5. **Monitor**: Track model performance in production

---

**Happy Auto Fine-Tuning! 🚀**
