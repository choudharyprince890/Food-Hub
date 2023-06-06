import os
import sys
import pandas as pd 
import numpy as np 
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# pipline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object
import warnings
warnings.filterwarnings("ignore")


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("src/artifcats/food_prediction","preprocessor.pkl")


#Create Data Transformation Class
class DataTransformationing:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_trainsformation_object(self):
        try:
            logging.info("Data Transformation Started")
            # Define which columns should be ordinal-encoded and which should be scaled

            num_features = ['prep_time']
            cat_features = ['cuisine', 'course', 'diet', 'contain_milk', 'contain_curd', 'onion/garlic']


            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('scaler',StandardScaler(with_mean=False))

                ]
            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('ordinalencoder',OneHotEncoder(sparse=False)),
                ('scaler',StandardScaler(with_mean=False))
                ]
            )

            #column transformer
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,num_features),
            ('cat_pipeline',cat_pipeline,cat_features)
            ])
            return preprocessor

        except Exception as e:
            logging.info("Error Occured In Data TRansformation Class")
            raise CustomException(e, sys)




    def start_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessor_obj = self.get_data_trainsformation_object()

            target_column_name = 'name'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            logging.info("after fit_transform")
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
