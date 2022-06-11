from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers
import pathlib
import PIL
import PIL.Image
import numpy as np

from data.components import read_imagefile,predict

import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__) 

app = FastAPI()
#model_dir = "my_model/1/"
#loaded_model = load_model('my_model/1/')

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

#class_names=['NG','OK']
@app.get("/")
async def root():
    return {"message": "Welcome to the iphone defect detection classification API!"}


@app.post("/predict/")
async def get_image_prediction_api(image: UploadFile = File(...)):
    """
    A function to transforms the raw image, loads saved model and returns predicted class

    :params image:upload a raw local image 
    :returns json format with keys 'model_prediction_class'
    """
    logger.info('predict_image POST request performed')
    logger.info('filename '+image.filename)

    image = read_imagefile(await image.read())
    response=predict(image)
    
    return response

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	run(app, host="0.0.0.0", port=port)