from sklearn.model_selection import train_test_split


class Validator:
    def __init__(self, args, kwargs):
        self.model = model(*args, **kwargs)

    def __train_test_split(self, data):
        self.data_train, self.data_test = train_test_split(data)

    def __fit(self):
        self.model.fit(self.data_train)

    def __predict(self):
        self.y_pred, self.y_test = self.model.predict(self.data_test)

    def __get_metrics(self):
        self.result = "some metrics"

    def run(self, data):
        self.__train_test_split(data)
        self.__fit()
        self.__predict()
        self.__get_metics()


        




