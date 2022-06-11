from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__) 

model_dir = "my_model/1/"
loaded_model = load_model('my_model/1/')
class_names=['NG','OK']

def read_imagefile(file):
    image = Image.open(BytesIO(file))
    return image

def predict(image):
    #image = read_imagefile(await image.read())
    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0
    predictions =loaded_model.predict(image)
    score = tf.nn.softmax(predictions[0])
    class_prediction = class_names[np.argmax(score)]

    return{
         "model_prediction_class": class_prediction,
         }

