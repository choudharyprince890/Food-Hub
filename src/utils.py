import os
import sys
import pickle

import pandas as pd 
import numpy as np 

from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from tensorflow import keras
import tensorflow as tf



## Save pickel File
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)





def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)




# this function will classify the dish in the image
def load_image_detection_model(model_path,file_path):
    try:
        classes = ['burger','butter_naan','chai', 'chapati','chole_bhature','dal_makhani','dhokla','fried_rice','idli','jalebi','kaathi_rolls','kadai_paneer','kulfi','masala_dosa','momos','paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa']
        model = keras.models.load_model(model_path)
        IMG_SIZE = (128, 128)
        raw_img = keras.preprocessing.image.load_img(file_path, target_size=IMG_SIZE)

        # Conver to to numpy array
        img_array = keras.preprocessing.image.img_to_array(raw_img)

        # Reshaping
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        # Make predictions
        predictions = model.predict(img_array)
        series = pd.Series(predictions[0], index=classes)

        proba = np.max(predictions)
        pred_class = classes[np.argmax(predictions)]

        logging.info("image is identified")
        return pred_class

    except Exception as e:
        raise CustomException(e, sys)
    