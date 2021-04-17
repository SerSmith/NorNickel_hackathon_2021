import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve

class ModelSick:
    def __init__(self, params, nrounds, seed=42 ):
        self.seed = seed
        self.models = {}
        self.trasholds = {}
        self.params = params
        self.nrounds = nrounds

    def fit(self, data, target):
        max_date = max(data.date)
        X_train = data[data.date < max_date]
        X_test = data[data.date == max_date]
        y_train = target[target.date < max_date]
        y_test = target[target.date == max_date]
        X_train = X_train.drop(['date', 'hash_tab_num'], axis = 1)
        X_test = X_test.drop(['date', 'hash_tab_num'], axis = 1)
        for i in range(1,13):
            y_col_name = 'y_' + str(i) 
            y_train_i = y_train[y_col_name]
            model = lgb.train(self.params[i]
                    , lgb.Dataset(X_train, label=y_train_i)
                    , num_boost_round=self.nrounds[i]
                    , verbose_eval=False)
            self.models[i] = model
            p, r, threshold = precision_recall_curve(y_test[y_col_name], self.models[i].predict(X_test))
            f1_scores = 2 * r * p / (r+p)
            f1_scores = f1_scores[p > 0]
            self.trasholds[i] = threshold[np.argmax(f1_scores)]
            print(f'f1_score_max = {np.max(f1_scores)}')

    def predict(self, data):
        predictions = pd.DataFrame()
        predictions[['hash_tab_num','date']] = data[['hash_tab_num','date']]
        X_data_predict = data.drop(['date', 'hash_tab_num'], axis = 1)
        for i in range(1,13):
            y_col_name = 'y_' + str(i)
            predictions[y_col_name] = (self.models[i].predict(X_data_predict) >= self.trasholds[i]).astype(int)
            #predictions[y_col_name] = self.models[i].predict(X_data_predict)

        return predictions