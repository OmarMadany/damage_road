from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil

app = FastAPI()
model = YOLO("lib/core/ai_server/best.pt")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # حفظ الصورة مؤقتًا
    with open("temp.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # تحليل الصورة
    results = model("temp.jpg")
    
    # تحويل النتائج لقائمة dict
    output = results.pandas().xyxy[0].to_dict(orient="records")
    return {"result": output}