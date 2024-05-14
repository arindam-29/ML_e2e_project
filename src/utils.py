import sys
import os
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import mean_squared_error, r2_score

from src.exception import CustomException

def save_object(file_path, save_obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            dill.dump(save_obj, file)
    
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(Xtr, ytr, Xte, yte, models):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            
            model.fit(Xtr, ytr)
            ytr_hat=model.predict(Xtr)
            yte_hat=model.predict(Xte)
            tr_score=r2_score(ytr, ytr_hat)
            te_score=r2_score(yte, yte_hat)

            report[list(models.keys())[i]]=te_score
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)