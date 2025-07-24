# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("dt_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

label_map = {0: "figurative", 1: "irony", 2: "regular", 3: "sarcasm"}

class MessageInput(BaseModel):
    text: str

@app.post("/predict/")
def predict_label(data: MessageInput):
    vector = vectorizer.transform([data.text])
    prediction = model.predict(vector)[0]
    label = label_map[int(prediction)]
    return {"prediction": label}
    