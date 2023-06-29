
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.component.data_ingestion import dataingestion
from src.component.data_transformation import DataTransform
from src.component.model_training import ModelTrainer


if __name__=='__main__':
    obj=dataingestion()
    train_data_path,test_data_path=obj.starting_data_ingestion()
    data_transformation = DataTransform()
    train_arr,test_arr,_=data_transformation.start_data_transform(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.start_model_training(train_arr,test_arr)




