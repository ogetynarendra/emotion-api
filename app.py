from fastapi import FastAPI, Request
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow access from other apps (like your Java backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

@app.post("/detect-emotion")
async def detect_emotion(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}
    result = classifier(text)
    return {"emotion": result[0]["label"]}
