from fastapi import FastAPI, Request
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a small but emotion-specific model
classifier = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-emotion")

@app.post("/detect-emotion")
async def detect_emotion(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}
    result = classifier(f"emotion: {text}")
    emotion = result[0]["generated_text"]
    return {"emotion": emotion}
