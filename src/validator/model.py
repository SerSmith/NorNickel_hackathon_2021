class TestModel:
    def __init__(self, **kwargs):
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

    def fit(self, data):
        pass

    def predict(self):
        return []