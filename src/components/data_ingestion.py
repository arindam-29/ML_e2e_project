import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformatio
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path:   str=os.path.join('artifacts', 'data.csv')
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path:  str=os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion - started")
        try:
            df=pd.read_csv('notebooks/data/stud.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train/Test split - started")
            train, test = train_test_split(df, test_size=0.2, random_state=1)
            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train/Test split - completed")
            logging.info("Data Ingestion - completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
# Testing

if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformatio()
    train_arr, test_arr, _=data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
