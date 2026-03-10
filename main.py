from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
import requests

app = FastAPI()
model = YOLO("best.pt")  # تأكد المسار صح

class ImageData(BaseModel):
    user_id: str
    image_url: str

@app.post("/analyze")
async def analyze(data: ImageData):
    # تحميل الصورة من الرابط
    resp = requests.get(data.image_url)
    img_path = "temp.jpg"
    with open(img_path, "wb") as f:
        f.write(resp.content)

    # تحليل الصورة باستخدام YOLO
    results = model(img_path)
    
    # تحويل النتائج لقائمة بسيطة للـ JSON
    detections = []
    for result in results:
        for box in result.boxes.xyxy:
            detections.append(box.tolist())
    
    return {
        "user_id": data.user_id,
        "detections": detections
    }
