from models.model import ModelSick
from validator.validator import Validator


def main():
    validator = Validator(ModelSick(), **{"X": None, "y": None})
    validator.train_test_split()
    validator.fit()
    validator.predict()


if __name__ == '__main__':
    main()