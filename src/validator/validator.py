import pandas as pd 
import numpy as np
from sklearn.metrics import f1_score


class Validator:
    def __init__(self, model, points,  *args, **kwargs):
        self.model = model(args, kwargs)
        self.points = points
        self.prediction = []
        self.f1_scores = []


    @staticmethod
    def add_one_year(d):
        return d + pd.to_timedelta(1, unit='y')

    
    def __drop_na(self):
        self.y.dropna(inplace=True)
        self.X = self.X.loc[self.y.index]
        self.X['date'] = pd.to_datetime(self.X['date'])

    
    def __train_test_split(self, point):
        train_indices = self.X[self.X.date < point].index
        test_indices = self.X[(self.X.date >= point) & (self.X.date < self.add_one_year(point))].index
        self.X_train = self.X.loc[train_indices]
        self.X_test = self.X.loc[test_indices]
        self.y_train = self.y.loc[train_indices]
        self.y_test = self.y.loc[test_indices]
        print(f"For point {point} train size: {self.X_train.shape[0]}, test size: {self.X_test.shape[0]}")

    def __fit(self):
        self.model.fit(self.X_train, self.y_train)

    def __predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def __get_metrics(self):
        self.prediction.append({"y_test": self.y_test, "y_pred": self.y_pred})

    def __calc_f1_score(self):
        for elem in self.prediction:
            f1 = []
            for i in range(1, 13):
                col_name = f"y_{i}"
                f1.append(f1_score(elem["y_test"][col_name], elem["y_pred"][col_name]))
            self.f1_scores.append(f1)
        self.f1_scores = np.array(self.f1_scores)
        self.f1_mean_scores = self.f1_scores.mean(axis=0)


    def run(self, X, y):
        self.X = X
        self.y = y
        self.__drop_na()
        for point in self.points:
            self.__train_test_split(point)
            self.__fit()
            self.__predict()
            self.__get_metrics()
        self.__calc_f1_score()


        




