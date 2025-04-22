import os
import uuid
import random
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from urllib.request import urlretrieve
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFaceHub
from ultralytics import YOLO

# Load environment variables
load_dotenv()
HuggingFaceHub_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HuggingFaceHub_API_KEY:
    raise EnvironmentError("Missing HUGGINGFACEHUB_API_TOKEN in environment.")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise EnvironmentError("Missing DEEPSEEK_API_KEY in environment.")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HuggingFaceHub_API_KEY
os.environ['DEEPSEEK_API_KEY'] = DEEPSEEK_API_KEY

# Initialize YOLO model
best_model = YOLO("best_yolov8_model.pt")  # Use relative path for deployment

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Mount static folder for audio
if not os.path.exists("audio"):
    os.makedirs("audio")
app.mount("/audio", StaticFiles(directory="audio"), name="audio")

# Pydantic input model
class ImageURL(BaseModel):
    url: str

# Function to classify image
def classify_piece(image_path):
    results = best_model(image_path)
    for result in results:
        class_ids = result.boxes.cls
        class_labels = [best_model.names[int(cls)] for cls in class_ids]
        final_output = class_labels[0]
        if final_output == "Green Head":
            return "The Berlin Green Head is an ancient Egyptian statue head"
        return final_output

# Generate short fact using Deepseek

def generate_facts(item):
    prompt = f"""
You are a Tour Guide That tells some facts about the {item}.
Create a short, engaging, and keyword-rich story for {item} based on the provided context.
Strictly adhere to 30–50 words without markdowns.
"""

    url = "https://api.deepseek.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful and creative assistant."},
            {"role": "user", "content": prompt.strip()}
        ],
        "temperature": round(random.uniform(0.9, 1.0), 2),
        "max_tokens": 100
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        return reply.strip()
    else:
        raise Exception(f"DeepSeek API Error: {response.status_code} - {response.text}")


# Text-to-speech function
def text_to_speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HuggingFaceHub_API_KEY}"}
    payload = {"inputs": message}

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())

    # Save MP3 audio
    mp3_file_path = "audio/audio.mp3"
    with open(mp3_file_path, "wb") as f:
        f.write(response.content)


@app.post("/museum/url/")
async def night_at_the_museum_url(data: ImageURL):
    try:
        print("[INFO] Starting image processing...")

        # Step 1: Download the image
        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
        print(f"[INFO] Downloading image from URL: {data.url}")
        urlretrieve(data.url, temp_filename)
        print(f"[INFO] Image downloaded and saved to: {temp_filename}")

        # Step 2: Classify the piece
        print("[INFO] Classifying the image...")
        item = classify_piece(temp_filename)
        print(f"[INFO] Classification result: {item}")

        # Step 3: Generate fact
        print("[INFO] Generating fact...")
        fact = generate_facts(item)
        print(f"[INFO] Generated fact: {fact}")

        # Step 4: Generate audio
        print("[INFO] Converting fact to speech...")
        text_to_speech(fact)
        print("[INFO] Audio saved successfully.")

        # Cleanup
        os.remove(temp_filename)
        print("[INFO] Temporary image deleted.")

        return {
            "artifact": item,
            "fact": fact,
            "audio_path": "audio/audio.mp3"
        }

    except Exception as e:
        print(f"[ERROR] An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

