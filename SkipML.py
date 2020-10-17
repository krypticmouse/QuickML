import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score

class SkipML:
    def __init__(self):
        self.model = None

    def fetch_model(self,df):
        pass

    def fit(self,x_train,y_train): 
        model_data =  pd.DataFrame(columns=['Model','Accuracy','F1 Score'])
        log_reg = LogisticRegression()
        log_reg.fit(x_train,y_train)
        lr_y_pred = log_reg.predict(x_train)
        lr_acc = accuracy_score(y_train,lr_y_pred)
        #lr_f1 = f1_score(y_train,lr_y_pred)
        print(lr_acc)
        self.model = log_reg
    
    def predict(self,X):
        return self.model.predict(X)
