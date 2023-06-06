import os
import sys

import pandas as pd 
import numpy as np 

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


from src.food_prediction.components.data_transformation import DataTransformationing


# from src.food_prediction.components.model_training import ModelTraningConfig
from src.food_prediction.components.model_training import ModelTraning



## initalising Data Ingestion
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("src/artifcats/food_prediction","train.csv")
    test_data_path:str = os.path.join("src/artifcats/food_prediction","test.csv")
    raw_data_path:str = os.path.join("src/artifcats/food_prediction","raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    ## Data Ingestion Method
    def initiated_data_ingestion(self):
        logging.info("Data Ingestion Method Started")
        try:
            logging.info("Before Data Ingestion ")
            print("before load daat file")
            data = pd.read_csv(os.path.join("data","food_prediction_cleaned_data_again2.csv"))
            # data = pd.read_csv("../data/food_prediction_cleaned_data_again2.csv")
            
            print("after load daat file")

            logging.info("Data Reading As Panda DataFraom")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Train Test Split")
            train_set,test_set = train_test_split(data,test_size=0.20,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("data ingestion Complites")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Error Occured in Data Ingestion Stage")
            raise CustomException(e, sys)


# Run
if __name__=="__main__":
    # data ingestion
    obj = DataIngestion()
    # obj.initiated_data_ingestion()
    train_data,test_data = obj.initiated_data_ingestion()
   

    # data transformation
    data_transformation = DataTransformationing()
    train_arr,test_arr,_ = data_transformation.start_data_transformation(train_data,test_data)


    modeltrainer=ModelTraning()
    print(modeltrainer.initatied_model_traning(train_arr,test_arr))
