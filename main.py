from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
import requests
import numpy as np
import cv2

app = FastAPI()
model = YOLO("best.pt")

class ImageData(BaseModel):
    user_id: str
    image_url: str

@app.post("/analyze")
async def analyze(data: ImageData):
    # تحميل الصورة من Cloudinary
    resp = requests.get(data.image_url)
    arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # تحليل الصورة
    results = model(img)
    output = results.pandas().xyxy[0].to_dict(orient="records")
    
    return {"user_id": data.user_id, "result": output}
