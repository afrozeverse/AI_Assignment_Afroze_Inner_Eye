import streamlit as st
import pickle
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

# Streamlit UI
st.set_page_config(page_title="Ticket Classifier", layout="centered")
st.title("🎫 Customer Support Ticket Classifier")
st.write("Enter a ticket description to predict its category")

# Input
ticket_description = st.text_area("Enter Ticket Description:", placeholder="e.g., I cannot login to the application", height=100)

# Predict button
if st.button("Predict Category"):
    if ticket_description.strip():
        cleaned = clean_text(ticket_description)
        
        # Prediction
        category = model.predict([cleaned])[0]
        confidence = max(model.predict_proba([cleaned])[0]) * 100
        
        # Display results
        st.success(f"**Predicted Category:** {category}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        
        # Show all probabilities
        classes = model.classes_
        probs = model.predict_proba([cleaned])[0]
        
        st.write("**All Probabilities:**")
        for cls, prob in zip(classes, probs):
            st.write(f"- {cls}: {prob*100:.2f}%")
    else:
        st.warning("Please enter a ticket description!")

# Footer
st.divider()
st.caption("Built with Python, Scikit-learn, and Streamlit")