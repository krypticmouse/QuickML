import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

class SkipML:
    def __init__(self):
        self.model = None

    def fit(self,x_train,y_train): 
        model_data =  pd.DataFrame(columns=['Model','Accuracy'])
        model_list = []
        best_idx = 0
        
        log_reg = LogisticRegression()
        log_reg.fit(x_train,y_train)
        lr_y_pred = log_reg.predict(x_train)
        lr_acc = accuracy_score(y_train,lr_y_pred)
        model_list.append(log_reg)
        model_data = model_data.append({'Model':'Linear Regression','Accuracy':lr_acc},ignore_index=True)
        
        
        dec_tree = DecisionTreeClassifier()
        dec_tree.fit(x_train,y_train)
        dt_y_pred = dec_tree.predict(x_train)
        dt_acc = accuracy_score(y_train,dt_y_pred)
        model_list.append(dec_tree)
        model_data = model_data.append({'Model':'Decision Tree Classifier','Accuracy':dt_acc},ignore_index=True)
        
        ran_for = RandomForestClassifier()
        ran_for.fit(x_train,y_train)
        rf_y_pred = ran_for.predict(x_train)
        rf_acc = accuracy_score(y_train,rf_y_pred)
        model_list.append(ran_for)
        model_data = model_data.append({'Model':'Random Forest Classifier','Accuracy':rf_acc},ignore_index=True)
        
        grad_boost = GradientBoostingClassifier()
        grad_boost.fit(x_train,y_train)
        gd_y_pred = grad_boost.predict(x_train)
        gd_acc = accuracy_score(y_train,gd_y_pred)
        model_list.append(ran_for)
        model_data = model_data.append({'Model':'Gradient Boosting Classifier','Accuracy':gd_acc},ignore_index=True)
        
        best_idx = np.argmax(model_data.Accuracy)
        self.model = model_list[best_idx]
        print(model_data.sort_values('Accuracy',ascending = False,ignore_index = True))

    def predict(self,X):
        return self.model.predict(X)