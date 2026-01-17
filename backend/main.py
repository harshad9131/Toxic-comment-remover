from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Labels must match training order
LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# Load model and vectorizer ONCE at startup
model = joblib.load("toxic_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI(title="YT Toxic Comment Detector")

class CommentRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "Backend running successfully"}

@app.post("/predict")
def predict(req: CommentRequest):
    # Vectorize input
    vec = vectorizer.transform([req.text])

    # Get probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
    else:
        scores = model.decision_function(vec)[0]
        probs = 1 / (1 + np.exp(-scores))  

    # Build response
    result = {
        label: float(prob)
        for label, prob in zip(LABELS, probs)
    }

    return {
        "text": req.text,
        "predictions": result
    }
