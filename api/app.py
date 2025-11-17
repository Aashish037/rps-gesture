from fastapi import FastAPI, UploadFile, File
import shutil
import os

# absolute import
from model.inference import predict_image

app = FastAPI()

UPLOAD_DIR = "./api/uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Gesture Model API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = predict_image(file_path)

    return {
        "prediction": prediction
    }
