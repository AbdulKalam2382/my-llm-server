from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi.responses import HTMLResponse
import torch

app = FastAPI(title="My LLM Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "./models/flan-t5-base"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded! Server is ready!")

class TextRequest(BaseModel):
    text: str
    max_length: int = 200

@app.get("/")
def root():
    return {"message": "LLM Server is running!", "endpoints": ["/summarize", "/write", "/health"]}

@app.get("/health")
def health():
    return {"status": "ok", "model": "flan-t5-base"}

@app.post("/summarize")
def summarize(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    prompt = f"summarize: {req.text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=req.max_length, num_beams=4, early_stopping=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"summary": result}

@app.post("/write")
def write(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=req.max_length, do_sample=True, temperature=0.7, top_p=0.9)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"output": result}
# Add this at the top with your other imports

# Add this route anywhere after the app is created
@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    with open("ui.html", encoding="utf-8") as f:
        return f.read()