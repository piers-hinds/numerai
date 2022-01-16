import pytest
import numpy as np
import pandas as pd
from nmr import *


@pytest.fixture
def simple_data():
    return pd.DataFrame({'era': np.concatenate([np.ones(1),
                                                np.ones(2)*2,
                                                np.ones(1)*3,
                                                np.ones(3)*4,
                                                np.ones(2)*5,
                                                np.ones(1)*6]).astype(int),
                        'value': np.random.random([10])})


def test_time_series_split_groups(simple_data):
    splitter = TimeSeriesSplitGroups(3)
    inds = []
    for train_index, test_index in splitter.split(simple_data, simple_data.value, simple_data.era):
        inds.append(train_index)
    assert (inds[0] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
    assert (inds[1] == np.array([0, 1, 2, 3, 4, 5, 6])).all()
    assert (inds[2] == np.array([0, 1, 2, 3])).all()


