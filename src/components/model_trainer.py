import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, evaluate_models_with_params

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("get train and test data")
            X_tr, y_tr, X_te, y_te=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info("model training - started")
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-N Classifier": KNeighborsRegressor(),
                "XGB Classifier": XGBRegressor(),
                "CatBoost Classifier":CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            #added for hyper parameter tuning
            params={
                "Random Forest":{
                    'n_estimators':[8, 16, 32, 64, 128, 256]
                },
                "Decision Tree":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Gradient Boosting":{
                    'learning_rate':[.1, .01, .05, .001],
                    'subsample':[.6, .7, .75, .8, .85, .9],
                    'n_estimators':[8, 16, 32, 64, 128, 256]
                },
                "Linear Regression":{},
                "K-N Classifier":{
                    'n_neighbors':[5, 7, 9, 11]
                },
                "XGB Classifier":{
                    'learning_rate':[.1, .01, .05, .001],
                    'n_estimators':[8, 16, 32, 64, 128, 256] 
                },
                "CatBoost Classifier":{
                    'learning_rate':[.1, .01, .05, .001],
                    'depth':[6, 8, 10],
                    'iterations':[30, 50, 100]
                },
                "AdaBoost Classifier":{
                    'learning_rate':[.1, .01, .05, .001],
                    'n_estimators':[8, 16, 32, 64, 128, 256]   
                }
            }

            #model_report:dict=evaluate_models(X_tr, y_tr, X_te, y_te, models)
            model_report:dict=evaluate_models_with_params(X_tr, y_tr, X_te, y_te, models, params)
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            logging.info("model training - completed")
            
            logging.info(f"saving best model {best_model_name}")
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                save_obj=best_model
            )

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)