
from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")

HF_URL = "https://router.huggingface.co/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

app = FastAPI()


class ChatRequest(BaseModel):
    text: str


@app.post("/chat")
def chat(req: ChatRequest):
    logger.info("POST /chat called")
    logger.info(f"User text: {req.text}")

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": req.text}
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }

    response = requests.post(
        HF_URL,
        headers=HEADERS,
        json=payload,
        timeout=60
    )

    logger.info(f"HF status: {response.status_code}")
    logger.info(f"HF raw response: {response.text}")

    if response.status_code != 200:
        return {
            "error": response.text,
            "status": response.status_code
        }

    data = response.json()

    return {
        "answer": data["choices"][0]["message"]["content"]
    }