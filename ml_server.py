from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load trained model and label encoder
model = joblib.load("news_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = FastAPI()

class NewsRequest(BaseModel):
    title: str
    description: str

@app.post("/predict")
def predict_category(news: NewsRequest):
    text = news.title + " " + news.description
    prediction = model.predict([text])[0]
    category = label_encoder.inverse_transform([prediction])[0]
    return {"category": category}

# Run this server using: uvicorn ml_server:app --reload
