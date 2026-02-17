from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

model = joblib.load("toxic_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI(title="YT Toxic Comment Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommentRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "Backend running successfully"}

@app.post("/predict")
def predict(req: CommentRequest):
    vec = vectorizer.transform([req.text])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
    else:
        scores = model.decision_function(vec)[0]
        probs = 1 / (1 + np.exp(-scores))

    result = {
        label: float(prob)
        for label, prob in zip(LABELS, probs)
    }

    return {
        "text": req.text,
        "predictions": result
    }
