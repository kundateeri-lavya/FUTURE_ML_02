"""
tests/test_classifier.py
========================
Basic unit tests for the Support Ticket Classifier.
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.classifier import TicketClassifier, TextPreprocessor, assign_priority


class TestTextPreprocessor:
    def setup_method(self):
        self.prep = TextPreprocessor()

    def test_clean_removes_urls(self):
        result = self.prep.clean("Visit https://example.com for help")
        assert 'http' not in result
        assert 'example' not in result

    def test_clean_lowercases(self):
        result = self.prep.clean("URGENT ISSUE")
        assert result == result.lower()

    def test_clean_removes_special_chars(self):
        result = self.prep.clean("Help! My app @crashed #bug")
        assert '!' not in result
        assert '@' not in result

    def test_preprocess_returns_string(self):
        result = self.prep.preprocess("The application is not working at all.")
        assert isinstance(result, str)
        assert len(result) > 0


class TestPriorityAssignment:
    def test_high_priority_urgent(self):
        assert assign_priority("URGENT: System is completely down!") == 'High'

    def test_high_priority_crash(self):
        assert assign_priority("App crash after update, cannot login.") == 'High'

    def test_medium_priority_slow(self):
        assert assign_priority("The dashboard is loading very slowly.") == 'Medium'

    def test_low_priority_general(self):
        assert assign_priority("Can you send me the user guide?") == 'Low'


class TestTicketClassifier:
    def setup_method(self):
        self.clf = TicketClassifier()
        # Train with minimal data
        texts = [
            "System crashed and I cannot login",
            "I was charged twice this month",
            "Please add dark mode feature",
            "How do I reset my password?",
            "The server is down and it's urgent!",
            "Invoice missing from my account",
        ]
        labels = [
            'Technical Issue',
            'Billing & Payment',
            'Feature Request',
            'Account Management',
            'Technical Issue',
            'Billing & Payment',
        ]
        self.clf.train(texts, labels)

    def test_predict_returns_dict(self):
        result = self.clf.predict("My app is crashing")
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self):
        result = self.clf.predict("I need a refund for my payment")
        assert 'category' in result
        assert 'priority' in result
        assert 'confidence' in result

    def test_predict_category_is_valid(self):
        result = self.clf.predict("Server is down, critical issue!")
        valid_categories = [
            'Technical Issue', 'Billing & Payment', 'Account Management',
            'Product Inquiry', 'Feature Request', 'Complaint', 'General Support'
        ]
        # Should be one of the trained categories
        assert result['category'] in ['Technical Issue', 'Billing & Payment',
                                       'Feature Request', 'Account Management']

    def test_predict_priority_is_valid(self):
        result = self.clf.predict("Some issue with my account")
        assert result['priority'] in ['High', 'Medium', 'Low']

    def test_untrained_raises_error(self):
        new_clf = TicketClassifier()
        with pytest.raises(RuntimeError):
            new_clf.predict("test ticket")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
