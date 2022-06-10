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
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__) 

app = FastAPI()
model_dir = "my_model/1/"
loaded_model = load_model('my_model/1/')

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

class_names=['NG','OK']
@app.get("/")
async def root():
    return {"message": "Welcome to the iphone defect detection classification API!"}

@app.post("/file/")
async def create_upload_file(image: UploadFile = File(...)):
    #logger.info('predict_image POST request performed')
    return {"filename": image.filename}

@app.post("/predict/")
async def get_image_prediction(image: UploadFile = File(...)):
    img_height = 224
    img_width = 224
    logger.info('predict_image POST request performed')
    logger.info('filename'+image.filename)
    pil_image = np.array(open(image.file))
    #img = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
    return {"filename": image.filename}
    # img = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
    # img_array = tf.keras.utils.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0)
    # predictions =loaded_model.predict(img_array)
    # score = tf.nn.softmax(predictions[0])
    # class_prediction = class_names[np.argmax(score)]
    # return{
    #     "model_prediction_class": class_prediction,
    #     }

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	run(app, host="0.0.0.0", port=port)