from fastapi import FastAPI, File, UploadFile
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os 

app = FastAPI()


MODEL = tf.keras.models.load_model('models/plants/2')

CLASS_NAMES = ['Pepper_bell_Bacterial_spot',
                'Pepper_bell_healthy',
                'Potato_Early_blight',
                'Potato_Late_blight',
                'Potato_healthy',
                'Tomato_Bacterial_spot',
                'Tomato_Early_blight',
                'Tomato_Late_blight',
                'Tomato_Leaf_Mold',
                'Tomato_Septoria_leaf_spot',
                'Tomato_Spider_mites_Two_spotted_spider_mite',
                'Tomato_Target_Spot',
                'Tomato_Tomato_YellowLeaf_Curl_Virus',
                'Tomato_Tomato_mosaic_virus',
                'Tomato_healthy']

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

    image_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(image_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]

    confidence = np.max(predictions[0])
    
    return {
        'predict': predicted_class,
        'confidence':  float(confidence),
        }


if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8800)