"""
predict.py — Interactive Support Ticket Classifier
===================================================
Run: python predict.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.classifier import TicketClassifier

PRIORITY_COLORS = {
    'High':   '\033[91m',  # red
    'Medium': '\033[93m',  # yellow
    'Low':    '\033[92m',  # green
}
RESET = '\033[0m'
BOLD  = '\033[1m'


def print_result(result: dict):
    p = result['priority']
    color = PRIORITY_COLORS.get(p, '')
    print(f"\n{'─'*55}")
    print(f"  {BOLD}Category :{RESET} {result['category']}")
    print(f"  {BOLD}Priority :{RESET} {color}{p}{RESET}")
    print(f"  {BOLD}Confidence:{RESET} {result['confidence']}")
    print(f"\n  {BOLD}All Category Scores:{RESET}")
    for cat, score in result['all_scores'].items():
        bar_len = int(float(score.rstrip('%')) / 5)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"    {cat:<22} {bar} {score}")
    print(f"{'─'*55}")


def main():
    model_path = 'models/classifier.pkl'
    if not os.path.exists(model_path):
        print("⚠️  Model not found. Training first...")
        import train
        train.main()

    print(f"\n{BOLD}{'='*55}")
    print("  🎫 Support Ticket Classifier — Interactive Mode")
    print(f"{'='*55}{RESET}")
    print("  Type a support ticket to classify it.")
    print("  Commands: 'quit' to exit | 'batch' for CSV mode\n")

    clf = TicketClassifier.load(model_path)
    print("✅ Model loaded successfully.\n")

    while True:
        try:
            ticket = input("📩 Enter ticket text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not ticket:
            continue
        if ticket.lower() == 'quit':
            print("Goodbye! 👋")
            break
        if ticket.lower() == 'batch':
            csv_path = input("  Enter CSV file path (must have a 'ticket_text' column): ").strip()
            if not os.path.exists(csv_path):
                print("  ❌ File not found.")
                continue
            import pandas as pd
            df = pd.read_csv(csv_path)
            if 'ticket_text' not in df.columns:
                print("  ❌ CSV must have a 'ticket_text' column.")
                continue
            results = [clf.predict(t) for t in df['ticket_text']]
            df['predicted_category'] = [r['category'] for r in results]
            df['predicted_priority']  = [r['priority']  for r in results]
            out = csv_path.replace('.csv', '_classified.csv')
            df.to_csv(out, index=False)
            print(f"  ✅ Results saved to {out}")
            continue

        result = clf.predict(ticket)
        print_result(result)


if __name__ == '__main__':
    main()
