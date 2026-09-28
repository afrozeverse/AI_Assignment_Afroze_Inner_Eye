import pandas as pd
import numpy as np
import re

def load_data(filepath):
    """Load CSV data"""
    df = pd.read_csv(filepath)
    return df

def clean_text(text):
    """Clean ticket description"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()  # Lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    
    return text

def preprocess_data(df):
    """Data preprocessing"""
    # Check missing values
    print("Missing values before handling:")
    print(df.isnull().sum())
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['ticket_description'], keep='first')
    print(f"Dataset shape after removing duplicates: {df.shape}")
    
    # Handle missing values in category
    df['category'] = df['category'].fillna('Other')
    
    # Clean ticket descriptions
    df['ticket_description'] = df['ticket_description'].apply(clean_text)
    
    print("\nDataset Info:")
    print(f"Total tickets: {len(df)}")
    print(f"Categories: {df['category'].unique()}")
    print(f"Sample cleaned text: {df['ticket_description'].iloc[0]}")
    
    return df

if __name__ == "__main__":
    df = load_data('data/tickets.csv')
    df_clean = preprocess_data(df)
    df_clean.to_csv('data/tickets_cleaned.csv', index=False)
    print("Cleaned data saved!")