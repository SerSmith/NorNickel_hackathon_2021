from model import TestModel
from validator import Validator


def main():
    validator = Validator(TestModel(), **{"X": None, "y": None})
    validator.train_test_split()
    validator.fit()
    validator.predict()


if __name__ == '__main__':
    main()