
from dataclasses import dataclass
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransform:
    def __init__(self):
        self.Data_Transfrom_config=DataTransformconfig()
        
    def get_data_transform_object(self):
        try:
            logging.info('Data transform initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            numerical_cols = ['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition','multiple_deliveries','Festival','Distance']
            categorical_cols = ['Weather_conditions','Road_traffic_density','Type_of_vehicle','City']
            
            Weather_conditions_categories =['Fog',"Stormy","Cloudy",'Sandstorms','Windy','Sunny']
            Road_traffic_density_categories =['Jam','High','Medium','Low']
            Type_of_vehicle_categories =['bicycle',"electric_scooter","scooter",'motorcycle']
            City_categories =['Semi-Urban',"Urban","Metropolitian"]
            
            logging.info('Pipeline Initiated')
            
            numeric_pip=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            categorical_pip=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinal',OrdinalEncoder(categories=[Weather_conditions_categories,Road_traffic_density_categories,Type_of_vehicle_categories,City_categories])),
                    ('scaler',StandardScaler())
                ]

            )
            ##pipline
            preprocessor=ColumnTransformer([
                ('numerical',numeric_pip,numerical_cols),
                ('categoris',categorical_pip,categorical_cols)
            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def start_data_transform(self,train_path,test_path):
        try:
            ## reading train and test data
            train_df=pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            
            preprocessing_obj =self.get_data_transform_object()
            
            target_column_name='Time_taken (min)'
            drop_columns=[target_column_name]
            
            input_features_train_df =train_df.drop(columns=drop_columns,axis=1)
            target_features_train_df =train_df[target_column_name]
            
            input_features_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_features_test_df=test_df[target_column_name]
            
             ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_features_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_features_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_features_test_df)]

            save_object(

                file_path=self.Data_Transfrom_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.Data_Transfrom_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)