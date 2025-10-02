from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model and vectorizer
model = joblib.load("log_reg_smote.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize FastAPI app
app = FastAPI(title="Telco Sentiment API")

# Input schema
class Review(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Welcome to the Telco Sentiment API 🚀"}

@app.post("/predict")
def predict_sentiment(review: Review):
    # Transform text
    features = vectorizer.transform([review.text])
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    sentiment = "Positive 😀" if prediction == 1 else "Negative 😠"

    return {
        "sentiment": sentiment,
        "confidence": round(float(prob), 3)
    }
