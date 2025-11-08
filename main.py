from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

model_name = "Obeida/smart_reply_v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

class Request(BaseModel):
    text: str

@app.post("/generate")
def generate_text(req: Request):
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return {"reply": tokenizer.decode(outputs[0], skip_special_tokens=True)}
