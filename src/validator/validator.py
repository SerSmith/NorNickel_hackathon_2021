from sklearn.model_selection import train_test_split


class Validator:
    def __init__(self, model, **kwargs):
        self.model = model
        for k in kwargs:
            self.__setattr__(k, kwargs[k])


    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)


    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def predict(self):
        self.y_pred = self.model.predict(self.X_test, self.y_test)


    def get_metrics(self):
        pass

        




