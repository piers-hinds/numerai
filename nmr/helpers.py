from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
import numpy as np
import pandas as pd
from numerapi import NumerAPI
from scipy.stats import spearmanr


class TimeSeriesSplitGroups(_BaseKFold):
    def __init__(self, n_splits=5):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_list = np.unique(groups)
        n_groups = len(group_list)
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_groups))
        indices = np.arange(n_samples)
        test_size = (n_groups // n_folds)
        test_starts = range(test_size + n_groups % n_folds,
                            n_groups, test_size)
        test_starts = list(test_starts)[::-1]
        for test_start in test_starts:
            yield (indices[groups.isin(group_list[:test_start])],
                   indices[groups.isin(group_list[test_start:test_start + test_size])])


def create_api(public_id=None, secret_key=None):
    """create a Numerai API instance
    if you want to submit your prediction,
    you need to create api key via "setting" of numerai webpage
    Args:
        public_id (str, optional): user id. Defaults to None.
        secret_key (str, optional): user api secret key. Defaults to None.
    Returns:
        NumerAPI: numerai api instance
    """
    if public_id == secret_key == None:
        return NumerAPI(verbosity="info")
    else:
        return NumerAPI(public_id, secret_key)


def load_data(fname='numerai_training_data.parquet'):
    data = pd.read_parquet(fname)
    data.dropna(inplace=True)
    features = data.columns[data.columns.str.startswith('feature')]
    return data, features


def spearman(y_true, y_pred):
    return spearmanr(y_pred, y_true).correlation
