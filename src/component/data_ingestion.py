import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from src.component.data_transformation import DataTransform

## create a file for ingestion
@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')
    
## creating class for  data ingestion
class dataingestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()
        
    def starting_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            '''firt need to read csv file'''
            df=pd.read_csv(os.path.join('notebook/data','cleand_finalTrain.csv'))
            logging.info('Dataset read')
            
            '''make a dir for raw data'''
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            '''adding raw data in dir'''
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('Ingestion of Data is completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)