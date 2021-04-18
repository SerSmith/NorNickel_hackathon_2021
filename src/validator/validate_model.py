from pprint import pprint
import warnings

import pandas as pd

from models.model import ModelSick
from validator.validator import Validator
from features import generate_features

warnings.filterwarnings('ignore')


def main():
    sot = pd.read_csv('data/sotrudniki.csv', sep = ';')
    rod = pd.read_csv('data/rodstvenniki.csv', sep = ';')
    ogrv = pd.read_csv('data/OGRV.csv', sep = ';')
    print("generate features")
    X, y = generate_features(sot, rod, ogrv)
    points = [
        pd.Timestamp(2018, 8, 1),
        pd.Timestamp(2018, 7, 1),
        pd.Timestamp(2017, 9, 1),
        pd.Timestamp(2017, 3, 1),
        pd.Timestamp(2016, 9, 1),
    ]
    validator = Validator(ModelSick, points)
    validator.run(X, y)
    pprint(validator.f1_scores)
    pprint(validator.f1_mean_scores)

if __name__ == '__main__':
    main()