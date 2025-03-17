import requests
import json
import pandas as pd
import re
import nltk
import textstat
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

nltk.download('punkt')

# Define readability-based age groups
def readability_to_age(text):
    score = textstat.flesch_kincaid_grade(text)
    if score <= 2:
        return "Kids (6-12)"
    elif 3 <= score <= 6:
        return "Teens (13-18)"
    elif 7 <= score <= 10:
        return "Young Adults (19-30)"
    elif 11 <= score <= 14:
        return "Adults (31-50)"
    else:
        return "Seniors (50+)"

# Function to fetch news data from API
def fetch_news(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch data")
        return {}

# Optimized function to assign age category using readability scoring
def assign_category(text):
    return readability_to_age(text)

# Main function to process news articles
def process_news(api_url, output_file="labeled_news.csv"):
    news_data = fetch_news(api_url)
    labeled_data = []
    
    for article in news_data.get("articles", []):
        title = article.get("title", "")
        description = article.get("description", "")
        combined_text = f"{title} {description}".strip()
        category = assign_category(combined_text)
        
        labeled_data.append({
            "title": title,
            "description": description,
            "category": category
        })
    
    df = pd.DataFrame(labeled_data)
    df.to_csv(output_file, index=False)
    print(f"Labeled data saved to {output_file}")
    return df

# Train a Decision Tree model and visualize it
def train_model(data):
    vectorizer = TfidfVectorizer()
    label_encoder = LabelEncoder()
    
    X = vectorizer.fit_transform(data['title'] + " " + data['description'])
    y = label_encoder.fit_transform(data['category'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = DecisionTreeClassifier(max_depth=5, class_weight="balanced")  # Handling class imbalance
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, predictions, target_names=label_encoder.classes_, labels=sorted(set(y_test))))
    
    # Visualize Decision Tree
    plt.figure(figsize=(15, 8))
    plot_tree(model, feature_names=vectorizer.get_feature_names_out(), class_names=label_encoder.classes_, filled=True, fontsize=6)
    plt.title("Decision Tree Visualization")
    plt.show()
    
    return make_pipeline(vectorizer, model), label_encoder

# Example usage
if __name__ == "__main__":
    API_KEY = "0ea2bdb2e0714ed0a010339f866ae4b0"
    API_URL = f"https://newsapi.org/v2/everything?q=all&apiKey={API_KEY}"
    labeled_data = process_news(API_URL)
    model_pipeline, label_enc = train_model(labeled_data)
