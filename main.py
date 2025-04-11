import os
import random
import shutil
import uuid
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFaceHub
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Check API key
if not API_KEY:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in .env")

# Load YOLO model once
best_model = YOLO("best_yolov8_model.pt")

# FastAPI initialization
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------- Helper Functions -----------

def classify_piece(image_path: str) -> str:
    results = best_model(image_path)
    for result in results:
        class_ids = result.boxes.cls
        class_labels = [best_model.names[int(cls)] for cls in class_ids]
        label = class_labels[0]
        return "The Berlin Green Head is an ancient Egyptian statue head" if label == "Green Head" else label

def generate_facts(item: str) -> str:
    template = """
    You are a Tour Guide that tells some facts about the {item}.
    Create a short, engaging, and keyword-rich story for {item} based on the provided context.
    Strictly adhere to 10–20 words.
    """
    prompt = PromptTemplate(template=template, input_variables=["item"])
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": random.uniform(0.9, 1), "max_length": 128}
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    description = chain.predict(item=item)
    return description.split('\n')[-1].strip()

def text_to_speech(message: str):
    api_url = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {"inputs": message}

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="TTS API request failed")

    with open("audio.flac", "wb") as f:
        f.write(response.content)

# ----------- FastAPI Endpoint -----------

@app.post("/museum/")
async def night_at_the_museum(image: UploadFile = File(...)):
    try:
        # Save uploaded image temporarily
        filename = f"temp_{uuid.uuid4().hex}.jpg"
        with open(filename, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Pipeline
        item = classify_piece(filename)
        fact = generate_facts(item)
        text_to_speech(fact)

        # Clean up
        os.remove(filename)

        return {"artifact": item, "fact": fact}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))