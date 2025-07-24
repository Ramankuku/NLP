from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("Decisions.pkl")
vectorizer = joblib.load("tfidf.pkl")

label_map = {0: "figurative", 1: "irony", 2: "regular", 3: "sarcasm"}

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    label = label_map[int(prediction)]
    return templates.TemplateResponse("form.html", {"request": request, "result": label})
