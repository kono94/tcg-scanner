# --- Old CLI test code preserved for reference ---
# from app.inferencer import classify_image
# from time import time
# from app.util import PROJECT_ROOT
# if __name__ == "__main__":
#     start = time()
#     predicted_class_name, confidence = classify_image(PROJECT_ROOT / 'resources' / 'test_images' / 'monkey.png')
#     inference_time_ms = (time() - start) * 1000  # Convert seconds to milliseconds
#     print(f"Inference time: {inference_time_ms:.2f} ms")
#     print(f"Predicted card ID: {predicted_class_name}, Confidence: {confidence:.4f}")

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import shutil
import os
from pathlib import Path
from app.inferencer import classify_image
from ultralytics import YOLO
import cv2
import numpy as np
import uuid

# Helper for price color
PRICE_COLOR = [
    (0.10, 'text-gray-500'),
    (0.5, 'text-yellow-500'),
    (3, 'text-orange-500'),
    (float('inf'), 'text-red-600'),
]
CONFIDENCE_COLOR = [
    (0.95, 'text-green-600'),
    (0.8, 'text-yellow-600'),
    (0.5, 'text-orange-600'),
    (0, 'text-red-600'),
]

def get_price_color(price):
    for threshold, color in PRICE_COLOR:
        if price < threshold:
            return color
    return 'text-gray-500'

def get_confidence_color(conf):
    for threshold, color in CONFIDENCE_COLOR:
        if conf > threshold:
            return color
    return 'text-red-600'

# Database setup
DATABASE_URL = "sqlite:///./cards.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Card(Base):
    __tablename__ = "cards"
    id = Column(Integer, primary_key=True, index=True)
    card_id = Column(String, unique=True, index=True)
    name = Column(String)
    price = Column(Float)
    image_url = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

# Load YOLO model once
YOLO_MODEL_PATH = Path("app/weights/card_detector.pt")
yolo_model = YOLO(str(YOLO_MODEL_PATH))

# FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Fake data for now
def seed_db():
    db = SessionLocal()
    if db.query(Card).count() == 0:
        db.add_all([
            Card(card_id="card001", name="Blue-Eyes White Dragon", price=12.34, image_url="/static/blue-eyes.jpg"),
            Card(card_id="card002", name="Dark Magician", price=8.99, image_url="/static/dark-magician.jpg"),
            Card(card_id="card003", name="Charizard", price=99.99, image_url="/static/charizard.jpg"),
        ])
        db.commit()
    db.close()
seed_db()

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
def upload_image(request: Request, file: UploadFile = File(...)):
    upload_dir = Path("app/static/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    # Generate a unique filename to avoid collisions
    ext = Path(file.filename).suffix or ".jpg"
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = upload_dir / unique_filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Read image for YOLO
    img = cv2.imread(str(file_path))
    results = yolo_model(img)
    db = SessionLocal()
    cards = []
    for box in results[0].boxes:
        b = box.xyxy[0].cpu().numpy().astype(int)
        crop = img[b[1]:b[3], b[0]:b[2]]
        card_id_path, confidence = classify_image(crop)

        if card_id_path is not None:
            card_id = os.path.splitext(card_id_path)[0]
            card = db.query(Card).filter(Card.card_id == card_id).first()
            cards.append({
                "card_id": card_id,
                "name": card.name,
                "price": card.price,
                "image_url": card.image_url if card.image_url else "/static/placeholder.jpg",
                "confidence": confidence,
                "price_color": get_price_color(card.price),
                "confidence_color": get_confidence_color(confidence),
            })
    db.close()
    # Render bundle template
    input_image_url = f"/static/uploads/{unique_filename}"
    return templates.TemplateResponse("bundle.html", {"request": request, "input_image_url": input_image_url, "cards": cards})

@app.get("/card/{card_id}", response_class=JSONResponse)
def get_card(card_id: str):
    db = SessionLocal()
    card = db.query(Card).filter(Card.card_id == card_id).first()
    db.close()
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    return {"card_id": card.card_id, "name": card.name, "price": card.price, "image_url": card.image_url}