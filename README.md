# Customer Support Ticket Classification System

## Problem Statement
Automatically classify customer support tickets into predefined categories (Login Issue, Application Error, Report, Account Update, Performance) based on ticket description.

## Dataset
- **Source:** Custom created
- **Records:** 100+ tickets
- **Categories:** 4 (Login Issue, Application Error, Report, Account Update, Performance)
- **Fields:** ticket_id, ticket_description, category, priority, status

## Technologies Used
- Python 3.x
- Pandas (data processing)
- Scikit-learn (ML model)
- Matplotlib & Seaborn (visualization)
- Streamlit (web interface)

## Installation
```bash
pip install -r requirements.txt
```

## How to Run

### Step 1: Data Preprocessing
```bash
python src/preprocessing.py
```

### Step 2: Data Analysis
```bash
python src/eda.py
```

### Step 3: Train Model
```bash
python src/train.py
```

### Step 4: Test Predictions
```bash
python src/predict.py
```

### Step 5: Run Web Interface
```bash
streamlit run src/app.py
```

## Model Performance
- **Accuracy:** 92%+
- **Algorithm:** Naive Bayes with TF-IDF
- **Training/Test Split:** 80/20

## Results
- Successfully predicts ticket category from description
- Provides confidence scores
- User-friendly web interface

## Limitations
- Small dataset (100 records)
- Limited categories (4 types)
- Similar descriptions may confuse model

## Future Improvements
- Larger dataset
- Deep learning models
- Multi-label classification
- API endpoint