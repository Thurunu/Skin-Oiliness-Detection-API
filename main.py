from io import BytesIO
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import cv2 as cv
import uvicorn

app = FastAPI()

MODEL = tf.keras.models.load_model("V 1.0.0.keras")
CLASS_NAMES = ['Acne', 'Dry', 'Normal', 'Oily']

@app.get("/ping")
async def ping():
    return "Server is running..."

@app.post("/level")
async def skin_level(file: UploadFile = File(...)):

    image = preprocess_image(await file.read())
    print('File read')


    prediction = MODEL.predict(np.expand_dims(image, axis=0))
    confidence = round(100 * (np.max(prediction[0])), 2)

    class_index = np.argmax(prediction)
    skin_type = CLASS_NAMES[class_index]

    return {
        'Skin Type': skin_type,
        'Confidence': confidence
    }

def preprocess_image(file):
    img_bytes = BytesIO(file)
    img = Image.open(img_bytes)
    img = np.array(img)
    
    resized = cv.resize(img, (224, 224))
    return resized


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)