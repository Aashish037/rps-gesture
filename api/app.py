import importlib
import os
import shutil
from typing import Callable

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

UPLOAD_DIR = "./api/uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

_predict_fn: Callable[[str], str] | None = None


def get_predict_fn() -> Callable[[str], str]:
    global _predict_fn
    if _predict_fn is None:
        inference = importlib.import_module("model.inference")
        _predict_fn = inference.predict_image
    return _predict_fn


@app.get("/")
def root():
    return {"message": "Gesture Model API is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = get_predict_fn()(file_path)

    return {
        "prediction": prediction
    }
