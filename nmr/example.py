from nmr.helpers import TimeSeriesSplitGroups
import numpy as np
import pandas as pd

x = np.random.random([10])
idx = np.linspace(0, 9, 10).astype(int)
groups = np.concatenate([np.ones(1), np.ones(2)*2, np.ones(1)*3, np.ones(3)*4, np.ones(2)*5,
                         np.ones(1)*6]).astype(int)
print(len(groups))
test_data = pd.DataFrame({'idx': idx, 'era': groups, 'value': x})


eras = test_data.era

splitter = TimeSeriesSplitGroups(3)
for i, (train_index, test_index) in enumerate(splitter.split(test_data, x, eras)):
    print('Fold: ', i)
    print(train_index); print(test_index)

