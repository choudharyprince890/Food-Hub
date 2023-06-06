import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("src/artifcats/food_prediction","model.pkl")
            preprocessor_path=os.path.join('src/artifcats/food_prediction','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)




class CustomData:
    def __init__(  self,
        cuisine: str,
        course: str,
        diet: str,
        prep_time: int,
        contain_milk: int,
        contain_curd: int,
        onion_garlic: int):

        self.cuisine = cuisine

        self.course = course

        self.diet = diet

        self.prep_time = prep_time

        self.contain_milk = contain_milk

        self.contain_curd = contain_curd

        self.onion_garlic = onion_garlic
        # print(type(" cuisine="+self.cuisine,"course="+self.course, "diet="+self.diet,"prep time=",self.prep_time,"contain milk="+self.contain_milk,"contain curd="+self.contain_curd,"onion garlic="+self.onion_garlic))
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "cuisine": [self.cuisine],
                "course": [self.course],
                "diet": [self.diet],
                "prep_time": [self.prep_time],
                "contain_milk": [self.contain_milk],
                "contain_curd": [self.contain_curd],
                "onion/garlic": [self.onion_garlic],
            }



            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)