# 🎫 Support Ticket Classification System

> Automatically classify customer support tickets into categories and assign priority levels using NLP and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

This project builds an end-to-end **NLP classification system** that:
- **Cleans and tokenizes** raw support ticket text
- **Classifies** tickets into 7 categories using TF-IDF + Logistic Regression
- **Assigns priority** (High / Medium / Low) using keyword-based logic
- **Evaluates** model performance with accuracy, F1 score, and confusion matrix
- **Supports batch processing** of CSV files

---

## 🗂 Project Structure

```
support-ticket-classifier/
├── data/
│   └── tickets.csv              # Auto-generated training dataset
├── models/
│   ├── classifier.pkl           # Saved trained model
│   ├── confusion_matrix.png     # Evaluation heatmap
│   └── priority_distribution.png
├── notebooks/
│   └── EDA_and_Training.ipynb  # Jupyter notebook walkthrough
├── src/
│   ├── __init__.py
│   ├── classifier.py            # Core ML pipeline
│   └── dataset.py               # Synthetic dataset generator
├── tests/
│   └── test_classifier.py       # Unit tests (pytest)
├── train.py                     # Training script
├── predict.py                   # Interactive CLI prediction
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/support-ticket-classifier.git
cd support-ticket-classifier
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Train the model
```bash
python train.py
```
This will:
- Generate the training dataset
- Train the TF-IDF + Logistic Regression model
- Print the classification report
- Save plots and the model to `models/`

### Step 2 — Predict interactively
```bash
python predict.py
```
Type any ticket text and get instant category + priority predictions.

### Step 3 — Batch classify from CSV
Inside `predict.py`, type `batch` and provide a CSV path with a `ticket_text` column.

### Step 4 — Jupyter Notebook (EDA)
```bash
jupyter notebook notebooks/EDA_and_Training.ipynb
```

### Step 5 — Run tests
```bash
python -m pytest tests/ -v
```

---

## 🧠 ML Pipeline

```
Raw Ticket Text
     ↓
Text Cleaning  (lowercase, remove URLs, special chars)
     ↓
Tokenization + Lemmatization (NLTK WordNetLemmatizer)
     ↓
TF-IDF Vectorization (unigrams + bigrams, max 5000 features)
     ↓
Logistic Regression Classifier
     ↓
Category Prediction + Confidence Score
     +
Priority Assignment (keyword rule engine)
```

---

## 🏷 Categories

| Category | Description |
|----------|-------------|
| Technical Issue | Crashes, errors, outages, login failures |
| Billing & Payment | Charges, refunds, invoices |
| Account Management | Password reset, profile, permissions |
| Product Inquiry | Plan questions, feature availability |
| Feature Request | New feature suggestions |
| Complaint | Service complaints, escalations |
| General Support | How-to questions, getting started |

## 🚦 Priority Levels

| Priority | Trigger Keywords |
|----------|-----------------|
| 🔴 High | urgent, critical, down, outage, crash, error, data loss |
| 🟡 Medium | slow, issue, problem, wrong, missing, glitch, bug |
| 🟢 Low | Everything else |

---

## 📊 Sample Output

```
📩 Enter ticket text: URGENT: Production server is down, cannot login!

─────────────────────────────────────────────────────
  Category : Technical Issue
  Priority : High
  Confidence: 94.3%

  All Category Scores:
    Technical Issue        ████████████████░░░░ 94.3%
    Account Management     █░░░░░░░░░░░░░░░░░░░  2.1%
    General Support        ░░░░░░░░░░░░░░░░░░░░  1.8%
    ...
─────────────────────────────────────────────────────
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| NLTK | Tokenization, lemmatization, stopwords |
| scikit-learn | TF-IDF, Logistic Regression, evaluation |
| pandas | Data handling |
| matplotlib / seaborn | Visualizations |
| Jupyter | Interactive exploration |
| pytest | Unit testing |

---

## 📈 Skills Demonstrated

- ✅ Text cleaning & tokenization
- ✅ NLP feature extraction (TF-IDF)
- ✅ Multi-class classification
- ✅ Priority logic (rule-based + ML hybrid)
- ✅ Model evaluation (precision, recall, F1, confusion matrix)
- ✅ Modular, production-ready code structure
- ✅ Unit testing with pytest
- ✅ Batch prediction from CSV

---

## 👤 Author

**Lavya**  
B.Tech Computer Science Engineering  
Avanthi Institute of Engineering and Technology, Hyderabad  
[GitHub](https://github.com/YOUR_USERNAME) | [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

## 📄 License

This project is licensed under the MIT License.
