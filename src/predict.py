import pickle
import pandas as pd
import re

def clean_text(text):
    """Clean input text"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load model
with open('model/ticket_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_category(description):
    """Predict category for new ticket"""
    cleaned = clean_text(description)
    
    # Predict
    category = model.predict([cleaned])[0]
    
    # Confidence (probability)
    probabilities = model.predict_proba([cleaned])[0]
    confidence = max(probabilities) * 100
    
    return category, confidence

# Test with new tickets
test_tickets = [
    "I cannot login to my account",
    "The application is showing error",
    "Please generate a sales report",
    "I need to update my phone number",
    "The system is very slow today"
]

print("=== TESTING NEW TICKETS ===\n")
for ticket in test_tickets:
    category, confidence = predict_category(ticket)
    print(f"Ticket: {ticket}")
    print(f"Predicted Category: {category}")
    print(f"Confidence: {confidence:.2f}%\n")