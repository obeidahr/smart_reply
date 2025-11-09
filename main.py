# main.py
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

# Lazy load
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True
    )

@app.post("/smart_reply")
async def smart_reply(request: Request):
    data = await request.json()
    text = data.get("message", "")
    prompt = f"Write a short, polite reply to: {text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=40)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"reply": reply}

@app.get("/")
def home():
    return {"status": "running"}
