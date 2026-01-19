import pickle
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image

app = FastAPI()

with open("mnist_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("L").resize((28, 28))
    data = np.array(img).reshape(1, -1)
    pred = model.predict(data)
    return {"digit": int(pred[0])}
