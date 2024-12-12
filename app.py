from flask import Flask, request, jsonify
from gemini_ai import RAGModel
import pandas as pd
import pdfplumber
import re
import os

app = Flask(__name__)

rag_model = RAGModel(model_name="gemini/rag-model")  # Initialize the Gemini AI RAG model

custom_categories = {
    "groceries": "Food & Dining",
    "utilities": "Bills & Payments",
    "rent": "Housing",
    # Additional categories if necessary
}

thresholds = {
    "Food & Dining": 500,
    "Bills & Payments": 300,
    "Housing": 1200,
    # Other category thresholds
}


# Function to extract transactions from PDF
def extract_transactions_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        transactions = []
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split("\n")
            for line in lines:
                match = re.match(r"(\d{2}-\w{3}-\d{4}) (.+?) (-?\d+\.\d{2})", line)
                if match:
                    date, description, amount = match.groups()
                    transactions.append({"Date": date, "Description": description, "Amount": float(amount)})
        return pd.DataFrame(transactions)

# Function to extract transactions from CSV
def extract_transactions_from_csv(csv_path):
    return pd.read_csv(csv_path)

# Categorize transactions using the RAG model
def categorize_transactions(transaction_descriptions, custom_categories=None):
    results = []
    for description in transaction_descriptions:
        context = rag_model.retrieve_context(description)  # Retrieve relevant context
        prediction = rag_model.generate_prediction(description, context)  # Generate prediction
        category = prediction["category"] if prediction["category"] else "Uncategorized"
        if custom_categories:
            category = custom_categories.get(category, category)
        results.append({"description": description, "category": category})
    return pd.DataFrame(results)

# Generate spending summary
def generate_spending_summary(df):
    spending_summary = df.groupby('category')['amount'].sum().reset_index()
    return spending_summary

# Generate overspending alerts based on thresholds
def generate_overspending_alerts(spending_summary, thresholds):
    overspending_alerts = []
    for _, row in spending_summary.iterrows():
        category, amount = row['category'], row['amount']
        if amount > thresholds.get(category, float('inf')):
            overspending_alerts.append(f"Alert: Overspending in {category}. Limit: ${thresholds[category]}, Spent: ${amount}")
    return overspending_alerts

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files['file']
    custom_categories = request.args.get('customCategories', None)
    
    if file.filename.endswith(".csv"):
        df = extract_transactions_from_csv(file)
    elif file.filename.endswith(".pdf"):
        df = extract_transactions_from_pdf(file)

    if custom_categories:
        custom_categories = dict([c.split(":") for c in custom_categories.split(",")])
    
    predictions = categorize_transactions(df['Description'].tolist(), custom_categories)
    df['Category'] = predictions['category']
    
    spending_summary = generate_spending_summary(df)
    alerts = generate_overspending_alerts(spending_summary, thresholds)
    
    return jsonify({"spending_summary": spending_summary.to_dict(orient="records"), "alerts": alerts})

if __name__ == "__main__":
    app.run(debug=True)
