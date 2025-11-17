# RPS Gesture Recognition

This repository contains a simple Rock-Paper-Scissors (RPS) hand-gesture recognition project using a deep learning model and a small API to serve predictions.

Live API

- The prediction API is deployed and available at: https://rps-gesture-2.onrender.com

What we did

- Implemented a dataset structure under `model/dataset` for training images (`paper/`, `rock/`, `scissors/`).
- Added a training script at `model/train.py` to train the gesture recognition model and save it to `model/saved_model/model.pth`.
- Implemented inference utilities in `model/inference.py` to load the saved model and run predictions on new images.
- Created a small API in the `api/` folder (`api/app.py`) to accept requests and return model predictions. Dependencies for the API are listed in `api/requirements.txt`.

Repository structure

```
api/
    app.py                # FastAPI app for serving predictions
    requirements.txt      # API dependencies
    uploads/              # (runtime) uploaded images for prediction
model/
    dataset/              # training images organized by class
        paper/
        rock/
        scissors/
    train.py              # training script
    inference.py          # prediction helper
    saved_model/
        model.pth         # trained model weights
```

Quick start — run the API locally

1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate    # on Windows (Git Bash / bash.exe)
# or on Unix/macOS: source .venv/bin/activate
```

2. Install API dependencies and run the server:

```bash
pip install -r api/requirements.txt
python api/app.py
```

Notes

- The repo includes a pre-trained model at `model/saved_model/model.pth` so you can call the live API directly or run the `api` server locally.
- If you want to retrain, use `python model/train.py` — ensure `model/dataset` is populated with images for each class.



