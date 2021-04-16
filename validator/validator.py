class Validator:
    def __init__(self, model, **kwargs):
        self.model = model
        for k in kwargs:
            self.__setattr__(k, kwargs[k])


    def fit(self):
        self.model.fit(self.data)


    def predict(self):
        self.model.predict()


    def split_train_test(self):
        pass




