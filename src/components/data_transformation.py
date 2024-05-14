import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformatio:
    def __init__(self):
        self.data_trasformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            num_cols=["writing_score", "reading_score"]
            cat_cols=["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoding", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Column preprocessing - started")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_cols),
                    ("cat_pipeline", cat_pipeline, cat_cols)
                ]
            )
            logging.info("Column preprocessing - completed")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("train and test reading - completed")
            preprocessor_obj=self.get_data_transformation_object()

            target_col_name="math_score"
            train_target_df=train_df[target_col_name]
            train_feature_df=train_df.drop(columns=[target_col_name], axis=1)
            test_target_df=test_df[target_col_name]
            test_feature_df=test_df.drop(columns=[target_col_name], axis=1)

            logging.info("train and test preprocessing - completed")
            train_preprocessor=preprocessor_obj.fit_transform(train_feature_df)
            test_preprocessor=preprocessor_obj.fit_transform(test_feature_df)

            train_arr=np.c_[train_preprocessor, np.array(train_target_df)]
            test_arr=np.c_[test_preprocessor, np.array(test_target_df)]

            logging.info("save preprocessing objects")

            save_object(
                file_path=self.data_trasformation_config.preprocessor_file_path,
                save_obj=preprocessor_obj
            )

            return(train_arr, test_arr, self.data_trasformation_config.preprocessor_file_path)

        except Exception as e:
            raise CustomException(e, sys)
