from fastapi import FastAPI, File, UploadFile
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "http://localhost:8601/v1/models/plant_model:predict"

CLASS_NAMES = ['Bell Pepper - Bacterial Spot',
                'Bell Pepper - Healthy',
                'Potato - Early Blight',
                'Potato - Late Blight',
                'Potato - Healthy',
                'Tomato - Bacterial Spot',
                'Tomato - Early Blight',
                'Tomato - Late Blight',
                'Tomato - Leaf Mold',
                'Tomato - Septoria Leaf Epot',
                'Tomato - Two-Spotted Spider Mite',
                'Tomato - Target Spot',
                'Tomato - YellowLeaf Curl Virus',
                'Tomato - Mosaic Virus',
                'Tomato - Healthy']

@app.get('/ping')
async def ping():
    return "Hello, sever is alive"

target_size = (150,150)

def file_as_image(data):
    image = Image.open(BytesIO(data))
    image = np.array(image.resize(target_size, Image.LANCZOS))
    return image


@app.post('/predict')
async def predict(
        file: UploadFile = File(...)    
):
    image = file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predict_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return {
        "Class": predict_class,
        "Confidence": float(confidence)
    }

if __name__=='__main__':
    uvicorn.run(app, host='localhost', port=8800)