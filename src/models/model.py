import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

class ModelSick:
    def __init__(self, seed=42):
        self.seed = seed
        self.models = {}
        self.trasholds = {}

    def fit(self, data, target):
        X = data.drop(['date', 'hash_tab_num'], axis = 1)
        X.fillna(0, inplace=True)
        for i in range(1,13):
            y_col_name = 'y_' + str(i) 
            y = target[y_col_name]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True, stratify=y)
            self.models[i] = RandomForestClassifier()
            self.models[i].fit(X_train, y_train)
            p, r, threshold = precision_recall_curve(y_test, self.models[i].predict_proba(X_test)[:,1])
            f1_scores = 2 * r * p / (r+p)
            f1_scores = f1_scores[p > 0]
            self.trasholds[i] = threshold[np.argmax(f1_scores)]

    def predict(self, data):
        predictions = pd.DataFrame()
        predictions[['hash_tab_num','date']] = data[['hash_tab_num','date']]
        X_data_predict = data.drop(['date', 'hash_tab_num'], axis = 1)
        X_data_predict.fillna(0, inplace=True)
        for i in range(1,13):
            y_col_name = 'y_' + str(i)
            predictions[y_col_name] = (self.models[i].predict_proba(X_data_predict)[:,1] >= self.trasholds[i]).astype(int)

        return predictions