from io import BytesIO
import pandas as pd 
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

#import logging

#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
#logger = logging.getLogger(__name__) 

model_dir = "my_model/1/"
loaded_model = load_model('my_model/1/')
class_names=['NG','OK']

prediction_history_df=pd.DataFrame(columns=['filename','model_prediction_class'])

def read_imagefile(file):
    """
    A function to read uploaded valid image through API endpoint
    """
    image = Image.open(BytesIO(file))
    return image

def predict(image,filename):
    """
    A function to transform image size, process image, load model, do predictions, save results 

    :params image:byte format image (from read_imagefile function)
    :params filename:filename of the image
    :returns json format result
    """
    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0
    predictions =loaded_model.predict(image)
    score = tf.nn.softmax(predictions[0])
    class_prediction = class_names[np.argmax(score)]
    prediction_result={
         "filename":filename,
         "model_prediction_class": class_prediction,
         }
    
    save_history_results(prediction_result)
    return prediction_result

def save_history_results(result):
    """
    A function to save to database(as csv file)
    """
    
    global prediction_history_df
    prediction_history_df=prediction_history_df.append(result, ignore_index=True)
    prediction_history_df.to_csv('history.csv',index=False)
    
