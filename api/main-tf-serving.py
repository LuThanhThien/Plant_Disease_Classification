from fastapi import FastAPI, File, UploadFile
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "http://localhost:8605/v1/models/potato_model:predict"

CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@app.get('/ping')
async def ping():
    return "Hello, sever is alive"

def file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
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
        "Confidence": confidence
    }

if __name__=='__main__':
    uvicorn.run(app, host='localhost', port=8000)