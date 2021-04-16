from sklearn.model_selection import train_test_split


class Validator:
    def __init__(self, args, kwargs):
        self.model = model(*args, **kwargs)

    def __train_test_split(self, data):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(*data)

    def __fit(self):
        self.model.fit(X_train, y_train)

    def __predict(self):
        self.y_pred = self.model.predict(self.X_test, y_test)

    def __get_metrics(self):
        self.result = "some metrics"

    def run(self, data):
        self.__train_test_split(data)
        self.__fit()
        self.__predict()
        self.__get_metics()


        




