from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI(title="Flan-T5 Base API")

# Load model once at startup
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

class Request(BaseModel):
    text: str

@app.post("/generate")
def generate(req: Request):
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"reply": response}

@app.get("/")
def root():
    return {"message": "Flan-T5 Base API is running ✅"}
