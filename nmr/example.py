from nmr import *
import numpy as np
import pandas as pd

tiny_data = pd.read_csv('../tiny_data.csv')
features = tiny_data.columns[tiny_data.columns.str.startswith('feature')]
targets = ['target']

splitter = TimeSeriesSplitGroups(5)
for train_index, test_index in splitter.split(tiny_data, tiny_data.target, tiny_data.era):
    pass
print(train_index); print(test_index)

train_dl, val_dl = get_dataloaders(tiny_data, train_index, test_index, features, targets)
print(tiny_data.target.iloc[:25])
for x, y in train_dl:
    print(y)