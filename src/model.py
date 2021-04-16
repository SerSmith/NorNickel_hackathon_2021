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

    def fit(self, data):
        for i in range(1,13):
            y_col_name = 'y_' + str(i) 
            X = data.dropna(subset=[y_col_name])\
            .drop(['y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 
                'y_7', 'y_8', 'y_9', 'y_10', 'y_11', 'y_12',
                'date', 'hash_tab_num'], axis = 1)
            X.fillna(0, inplace=True)
            y = data.dropna(subset=[y_col_name])[y_col_name]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True, stratify=y)
            self.models[i] = RandomForestClassifier()
            self.models[i].fit(X_train, y_train)
            p, r, threshold = precision_recall_curve(y_test, self.models[i].predict_proba(X_test)[:,1])
            f1_scores = 2 * r * p / (r+p)
            f1_scores = f1_scores[p > 0]
            self.trasholds[i] = threshold[np.argmax(f1_scores)]

    def predict(self, data_predict):
        predictions = pd.DataFrame()
        predictions['hash_tab_num'] = data_predict['hash_tab_num']
        for i in range(1,13):
            y_col_name = 'y_' + str(i)
            X_data_predict = data_predict\
                    .drop(['y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 
                        'y_7', 'y_8', 'y_9', 'y_10', 'y_11', 'y_12',
                        'date', 'hash_tab_num'], axis = 1)

            X_data_predict.fillna(0, inplace=True)
            predictions[y_col_name] = (self.models[i].predict_proba(X_data_predict)[:,1] >= self.trasholds[i]).astype(int)

        return predictions