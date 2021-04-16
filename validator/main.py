from model import TestModel
from validator import Validator


def main():
    validator = Validator(TestModel(), **{"data": "test"})
    validator.fit()
    print(validator.predict())


if __name__ == '__main__':
    main()