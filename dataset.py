"""
Dataset Generator
=================
Generates a synthetic dataset of support tickets for training.
In a real project, replace this with your actual CSV/database data.
"""

import pandas as pd
import random

TICKETS = {
    'Technical Issue': [
        "My application keeps crashing every time I open it.",
        "The system is down and I cannot access my account.",
        "I'm getting a 500 internal server error on the login page.",
        "The software is not working after the latest update.",
        "I'm experiencing an outage and need urgent help.",
        "App crashes immediately after launch on Android.",
        "Cannot connect to the server, getting timeout errors.",
        "The dashboard is not loading at all.",
        "Error message: database connection failed.",
        "The export feature is broken, files won't download.",
        "API integration is throwing authentication errors.",
        "Screen freezes when I try to upload documents.",
        "Getting a blank white screen after login.",
        "Search functionality is completely broken.",
        "Push notifications stopped working suddenly.",
        "The mobile app crashes on iOS 17.",
        "Cannot install the software, setup fails at 80%.",
        "File upload giving 413 error, please fix urgently.",
        "Two-factor authentication not sending SMS codes.",
        "The report generation feature is stuck on loading.",
    ],
    'Billing & Payment': [
        "I was charged twice for my subscription this month.",
        "My payment failed but the amount was deducted from my account.",
        "I need an invoice for my last three payments.",
        "How do I update my credit card information?",
        "I cancelled my subscription but still got charged.",
        "Refund not received after 10 business days.",
        "I need to change my billing cycle from monthly to annual.",
        "The discount code I applied is not reflecting in my bill.",
        "My account shows past due even though I paid.",
        "Can I get a receipt for my payment made last week?",
        "Incorrect amount was deducted from my bank account.",
        "I want to downgrade my plan and get a partial refund.",
        "Tax invoice is missing my GST number.",
        "Payment gateway keeps rejecting my card.",
        "I need to add a purchase order number to my invoice.",
    ],
    'Account Management': [
        "I forgot my password and cannot reset it via email.",
        "I want to delete my account and all associated data.",
        "How do I transfer my account to a different email address?",
        "I am locked out of my account after too many login attempts.",
        "Need to add a secondary admin user to my organization.",
        "My account was suspended without any prior notice.",
        "How do I enable two-factor authentication on my account?",
        "I need to update my company name and address in my profile.",
        "Can I merge two accounts into one?",
        "How do I change my username?",
        "My account permissions were changed without my consent.",
        "I want to export all my account data.",
        "How to revoke access for a former employee?",
        "My profile picture is not updating.",
        "I need to change the timezone settings on my account.",
    ],
    'Product Inquiry': [
        "What are the differences between the Basic and Pro plans?",
        "Does your product support integration with Salesforce?",
        "How many users can I add to the Enterprise plan?",
        "Is there a free trial available for the premium features?",
        "What file formats are supported for data import?",
        "Do you offer a student or educational discount?",
        "Can I use the API with the basic plan?",
        "Does the mobile app work offline?",
        "What is the maximum storage limit on the Pro plan?",
        "Is the software HIPAA compliant for healthcare use?",
        "Do you support single sign-on (SSO) integration?",
        "How many API calls are allowed per month?",
        "Is there a white-label option available?",
        "Does the tool support multiple languages?",
        "What analytics and reporting features are available?",
    ],
    'Feature Request': [
        "It would be great to have a dark mode option in the app.",
        "Please add the ability to bulk export data to Excel.",
        "Can you add Slack integration to the notification settings?",
        "I would love a mobile app for Android tablets.",
        "Please add keyboard shortcuts for common actions.",
        "It would be helpful to have a calendar view for tasks.",
        "Can you add custom fields to the report builder?",
        "Please allow us to schedule automated reports via email.",
        "I'd like the ability to tag and filter tickets by custom labels.",
        "Can you add an undo button for accidental deletions?",
        "Please add support for CSV import of user data.",
        "Would love a Zapier integration for workflow automation.",
        "Can the dashboard widgets be rearranged by drag and drop?",
        "Please add a read receipt feature for messages.",
        "I'd like to see a Gantt chart view for project timelines.",
    ],
    'Complaint': [
        "Your customer support response time is unacceptably slow.",
        "I have been waiting for a resolution for over two weeks now.",
        "The last agent I spoke to was very rude and unhelpful.",
        "Your software is full of bugs and I am very disappointed.",
        "I am extremely frustrated with the lack of communication.",
        "This is the third time I am reporting the same issue.",
        "The service quality has degraded significantly this month.",
        "I feel misled by your marketing claims about the product.",
        "My issue was closed without being resolved.",
        "I demand a refund for the poor service I received.",
        "Your onboarding process is confusing and needs improvement.",
        "I was transferred between agents four times with no help.",
        "The product does not work as advertised.",
        "I am considering cancelling my subscription due to poor service.",
        "This is unacceptable — I need escalation to a manager.",
    ],
    'General Support': [
        "How do I get started with the product after signing up?",
        "Where can I find the documentation and user guides?",
        "Can you help me understand how the dashboard works?",
        "I need help setting up my first project.",
        "What are the system requirements to run your software?",
        "How do I contact support for urgent issues?",
        "Is there a community forum or knowledge base available?",
        "Can you walk me through how to generate my first report?",
        "How do I invite team members to my workspace?",
        "Where do I find my API keys?",
        "How do I configure email notifications?",
        "What is the best way to import my existing data?",
        "Can you explain what the usage limits mean?",
        "How do I set up webhooks for real-time updates?",
        "I need a demo of the advanced analytics features.",
    ],
}


def generate_dataset(samples_per_category: int = 30) -> pd.DataFrame:
    """Generate a synthetic dataset with augmented samples."""
    data = []
    random.seed(42)

    for category, templates in TICKETS.items():
        for _ in range(samples_per_category):
            text = random.choice(templates)
            # Simple augmentation: add urgency markers to ~20% of tickets
            if random.random() < 0.2:
                prefix = random.choice([
                    "URGENT: ", "Please help ASAP: ", "Critical issue: ", ""
                ])
                text = prefix + text
            data.append({
                'ticket_text': text,
                'category': category,
                'priority': assign_priority_for_dataset(text)
            })

    df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def assign_priority_for_dataset(text: str) -> str:
    from classifier import assign_priority
    return assign_priority(text)


if __name__ == "__main__":
    df = generate_dataset(samples_per_category=30)
    df.to_csv('tickets.csv', index=False)
    print(f"✅ Dataset saved: {len(df)} tickets across {df['category'].nunique()} categories")
    print(df['category'].value_counts())
    print(df['priority'].value_counts())
