import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve

class ModelSick:
    def __init__(self, params, nrounds, quantity_models, validation_period):
        self.models = []
        self.trasholds = {}
        self.params = params
        self.nrounds = nrounds
        self.quantity_models = quantity_models
        self.validation_period = validation_period

    def fit(self, data, target):

        for i in range(self.quantity_models):
            params = self.params
            params["random_state"] = i

            max_date = max(data.date)
            X_train = data[data.date < max_date - pd.DateOffset(months=3)]
            y_train = target[target.date < max_date - pd.DateOffset(months=3)]


            X_train = X_train.drop(['date', 'hash_tab_num'], axis=1)


            model_by_seed = {}

            for i in range(1, 13):
                y_col_name = 'y_' + str(i) 
                y_train_i = y_train[y_col_name]
                model = lgb.train(self.params[i]
                        , lgb.Dataset(X_train, label=y_train_i)
                        , num_boost_round=self.nrounds[i]
                        , verbose_eval=False)
                model_by_seed[i] = model

            self.models.append(model_by_seed)



    
        y_test_global = pd.DataFrame([], columns=['hash_tab_num', 'date', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 
                'y_7', 'y_8', 'y_9', 'y_10', 'y_11', 'y_12'])
        predict_global = pd.DataFrame([], columns=['hash_tab_num', 'date', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 
                'y_7', 'y_8', 'y_9', 'y_10', 'y_11', 'y_12'])

 

        for i in range(self.validation_period):
            X_test = data[data.date == max_date - pd.DateOffset(months=i)]
            y_test = target[target.date == max_date - pd.DateOffset(months=i)]

            y_test_global = y_test_global.append(y_test, ignore_index=True)
            predict_global = predict_global.append(self.predict_proba(X_test), ignore_index=True)


        for i in range(1, 13):
            y_col_name = 'y_' + str(i) 
            p, r, threshold = precision_recall_curve(y_test_global[y_col_name], predict_global[y_col_name])
            f1_scores = 2 * r * p / (r+p)
            f1_scores = f1_scores[p > 0]
            self.trasholds[i] = threshold[np.argmax(f1_scores)]
            print(f'f1_score_max = {np.max(f1_scores)}')

    def predict_proba(self, data):

        predictions = pd.DataFrame([], columns=['hash_tab_num', 'date', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 
                'y_7', 'y_8', 'y_9', 'y_10', 'y_11', 'y_12'])

        for model_by_seed in self.models:
            predictions_by_seed = pd.DataFrame()
            predictions_by_seed[['hash_tab_num', 'date']] = data[['hash_tab_num', 'date']]
            X_data_predict = data.drop(['date', 'hash_tab_num'], axis=1)

            for i in range(1, 13):
                y_col_name = 'y_' + str(i)
                predictions_by_seed[y_col_name] = model_by_seed[i].predict(X_data_predict)

            predictions = predictions.append(predictions_by_seed, ignore_index=True)
        predictions = predictions.groupby(['hash_tab_num', 'date']).mean().reset_index()

        return predictions

    def predict(self, data):

        predictions = self.predict_proba(data)
        for i in range(1, 13):
                y_col_name = 'y_' + str(i)
                predictions[y_col_name] = (predictions[y_col_name] >= self.trasholds[i]).astype(int)


        return predictions