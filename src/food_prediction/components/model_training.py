import os 
import sys
import pandas as pd 
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException




from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


from src.utils import save_object
from src.utils import evaluate_models



@dataclass
class ModelTraningConfig:
    trained_model_file_path = os.path.join("src/artifcats/food_prediction","model.pkl")



class ModelTraning:
    def __init__(self):
        self.model_trainer_config = ModelTraningConfig()

    def initatied_model_traning(self,train_array,test_array):
        try:
            logging.info("Split Dependent And Independent Features")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
            }
            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            ## To get best model value from models dictionary
            best_model_score = max(sorted(model_report.values()))

            ## To get best model key from models dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            
            logging.info("Started model training")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Model Training Compleated")

            predicted=best_model.predict(X_test)

            r2_square = accuracy_score(y_test, predicted)
            logging.info("R2 score Calculted")

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)