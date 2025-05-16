from fastapi import FastAPI, UploadFile, File
from app.predict import predict_trash

app = FastAPI(title="Trash Classifier API")

@app.post("/predict_image")
async def classify_image(file: UploadFile = File(...)):
    result = await predict_trash(file)
    return result

@app.get("/test")
async def testing():
    return 'Welcome to FastAPI'