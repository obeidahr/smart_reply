from fastapi import FastAPI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

app = FastAPI()

model_name = "google/flan-t5-small"  # lightweight model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

@app.get("/")
def home():
    return {"status": "Smart Reply model ready 🚀"}

@app.post("/reply")
def generate_reply(email: str):
    prompt = f"Write a short, polite email reply to: {email}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=60)
    return {"reply": tokenizer.decode(outputs[0], skip_special_tokens=True)}
