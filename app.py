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

# Load ML model (lighter one to fit in 512MB Render limit)
classifier = pipeline("text-classification", model="prajjwal1/bert-tiny", return_all_scores=False)

@app.post("/detect-emotion")
async def detect_emotion(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}
    result = classifier(text)
    return {"emotion": result[0]["label"]}
