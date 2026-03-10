from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO

app = FastAPI()
model = YOLO("best.pt")  # تأكد إن المسار صح

class ImageData(BaseModel):
    user_id: str
    image_url: str

@app.post("/analyze")
async def analyze(data: ImageData):
    # تحميل الصورة من الرابط
    resp = requests.get(data.image_url)
    img = Image.open(BytesIO(resp.content))

    # تصغير الصورة لتجنب وقت معالجة طويل
    img = img.resize((640, 640))
    img_path = "temp.jpg"
    img.save(img_path)

    # تحليل الصورة باستخدام YOLO
    results = model(img_path)

    # تحويل النتائج لقائمة بسيطة للـ JSON
    detections = []
    for result in results:
        if hasattr(result, "boxes") and result.boxes is not None:
            for box in result.boxes.xyxy:
                detections.append(box.tolist())

    return {
        "user_id": data.user_id,
        "detections": detections
    }
