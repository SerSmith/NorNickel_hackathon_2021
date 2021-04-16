import warnings

import pandas as pd

from models.model import ModelSick
from validator.validator import Validator
from features.features import generate_features

warnings.filterwarnings('ignore')

def main():
    sot = pd.read_csv('data/sotrudniki.csv', sep = ';')
    rod = pd.read_csv('data/rodstvenniki.csv', sep = ';')
    ogrv = pd.read_csv('data/OGRV.csv', sep = ';')
    data = generate_features(sot, rod, ogrv) 
    validator = Validator(ModelSick(), agrs, kwargs)
    validator.run(data)
    print(validator.result)

if __name__ == '__main__':
    main()