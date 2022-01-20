from nmr import *
import numpy as np
import pandas as pd

tiny_data = pd.read_csv('../small_data.csv')
features = tiny_data.columns[tiny_data.columns.str.startswith('feature')]
targets = ['target']
tiny_data

splitter = TimeSeriesSplitGroups(5)
new_splitter = PurgedSlidingSplit(3)

# for train_index, test_index in splitter.split(tiny_data, tiny_data.target, tiny_data.era):
#     print('Train: ', train_index);
#     print('Test: ', test_index)

for train_index, test_index in new_splitter.split(tiny_data, tiny_data.target, tiny_data.era):
    print('Train: ', train_index);
    print('Test: ', test_index)