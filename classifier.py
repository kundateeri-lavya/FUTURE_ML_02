"""
Support Ticket Classifier
=========================
Classifies customer support tickets into categories and assigns priority levels.
Tools: NLTK, scikit-learn, TF-IDF
"""

import re
import pickle
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
def download_nltk_data():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

download_nltk_data()


# ── Text Preprocessing ────────────────────────────────────────────────────────

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)          # remove URLs
        text = re.sub(r'[^a-z\s]', ' ', text)               # keep only letters
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_lemmatize(self, text: str) -> str:
        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 2
        ]
        return ' '.join(tokens)

    def preprocess(self, text: str) -> str:
        return self.tokenize_and_lemmatize(self.clean(text))


# ── Priority Logic ────────────────────────────────────────────────────────────

HIGH_PRIORITY_KEYWORDS = [
    'urgent', 'critical', 'down', 'outage', 'emergency', 'asap',
    'immediately', 'broken', 'crash', 'error', 'failure', 'not working',
    'cannot access', 'data loss', 'security breach', 'hacked', 'payment failed',
    'cannot login', 'system down', 'production issue'
]

MEDIUM_PRIORITY_KEYWORDS = [
    'slow', 'delay', 'issue', 'problem', 'trouble', 'incorrect',
    'wrong', 'missing', 'update', 'change', 'request', 'question',
    'not loading', 'glitch', 'bug'
]


def assign_priority(text: str) -> str:
    text_lower = text.lower()
    for kw in HIGH_PRIORITY_KEYWORDS:
        if kw in text_lower:
            return 'High'
    for kw in MEDIUM_PRIORITY_KEYWORDS:
        if kw in text_lower:
            return 'Medium'
    return 'Low'


# ── Ticket Category Classifier ────────────────────────────────────────────────

CATEGORIES = [
    'Technical Issue',
    'Billing & Payment',
    'Account Management',
    'Product Inquiry',
    'Feature Request',
    'Complaint',
    'General Support'
]


class TicketClassifier:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=1
            )),
            ('clf', LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42
            ))
        ])
        self.is_trained = False

    def train(self, X: list, y: list):
        X_clean = [self.preprocessor.preprocess(t) for t in X]
        self.pipeline.fit(X_clean, y)
        self.is_trained = True
        print("✅ Model trained successfully.")

    def predict(self, text: str) -> dict:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        clean = self.preprocessor.preprocess(text)
        category = self.pipeline.predict([clean])[0]
        proba = self.pipeline.predict_proba([clean])[0]
        classes = self.pipeline.classes_
        confidence = round(float(max(proba)) * 100, 2)
        priority = assign_priority(text)
        return {
            'ticket_text': text,
            'category': category,
            'priority': priority,
            'confidence': f"{confidence}%",
            'all_scores': {
                cls: f"{round(p*100, 1)}%"
                for cls, p in zip(classes, proba)
            }
        }

    def evaluate(self, X_test: list, y_test: list):
        X_clean = [self.preprocessor.preprocess(t) for t in X_test]
        y_pred = self.pipeline.predict(X_clean)
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        return y_pred

    def save(self, path: str = 'models/classifier.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Model saved to {path}")

    @staticmethod
    def load(path: str = 'models/classifier.pkl') -> 'TicketClassifier':
        with open(path, 'rb') as f:
            return pickle.load(f)
