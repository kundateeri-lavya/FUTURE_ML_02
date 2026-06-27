"""
train.py — Train and evaluate the Support Ticket Classifier
============================================================
Run: python train.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Make sure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.classifier import TicketClassifier
from src.dataset import generate_dataset


def plot_confusion_matrix(y_test, y_pred, labels, save_path='models/confusion_matrix.png'):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels
    )
    plt.title('Confusion Matrix — Support Ticket Classifier', fontsize=14)
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix saved to {save_path}")


def plot_priority_distribution(df, save_path='models/priority_distribution.png'):
    counts = df['priority'].value_counts()
    colors = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}
    plt.figure(figsize=(6, 4))
    bars = plt.bar(counts.index, counts.values,
                   color=[colors.get(p, '#95a5a6') for p in counts.index])
    plt.title('Ticket Priority Distribution')
    plt.ylabel('Count')
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Priority distribution saved to {save_path}")


def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # ── 1. Load or generate dataset ──────────────────────────────────────────
    csv_path = 'data/tickets.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"📂 Loaded existing dataset: {len(df)} tickets")
    else:
        print("🔄 Generating synthetic dataset...")
        df = generate_dataset(samples_per_category=30)
        df.to_csv(csv_path, index=False)
        print(f"✅ Dataset saved: {len(df)} tickets")

    print("\nCategory distribution:")
    print(df['category'].value_counts().to_string())
    print("\nPriority distribution:")
    print(df['priority'].value_counts().to_string())

    # ── 2. Train / Test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df['ticket_text'].tolist(),
        df['category'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['category']
    )
    print(f"\nTraining samples : {len(X_train)}")
    print(f"Testing  samples : {len(X_test)}")

    # ── 3. Train model ────────────────────────────────────────────────────────
    print("\n🚀 Training model...")
    clf = TicketClassifier()
    clf.train(X_train, y_train)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    y_pred = clf.evaluate(X_test, y_test)

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    labels = sorted(df['category'].unique().tolist())
    plot_confusion_matrix(y_test, y_pred, labels)
    plot_priority_distribution(df)

    # ── 6. Save model ─────────────────────────────────────────────────────────
    clf.save('models/classifier.pkl')

    # ── 7. Demo predictions ───────────────────────────────────────────────────
    demo_tickets = [
        "URGENT: System is completely down, no one can log in!",
        "I would like to request a dark mode feature.",
        "I was charged twice this month, please refund.",
        "How do I reset my password?",
        "The app keeps crashing on my iPhone.",
    ]
    print("\n🎯 Demo Predictions:")
    print("=" * 65)
    for ticket in demo_tickets:
        result = clf.predict(ticket)
        print(f"\nTicket   : {ticket[:60]}...")
        print(f"Category : {result['category']}")
        print(f"Priority : {result['priority']}")
        print(f"Confidence: {result['confidence']}")
    print("=" * 65)
    print("\n✅ Training complete! Model saved to models/classifier.pkl")


if __name__ == '__main__':
    main()
