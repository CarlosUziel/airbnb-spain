import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def cross_val_lgbm(
    data: pd.DataFrame, y: pd.Series, random_seed: int = 8080, n_jobs: int = 8
):
    """
    K-fold cross validation of an ElasticNet model

    Args:
        data: full input data.
        y: value targets of the dependent variable.
        random_seed: seed to set random state.
    """
    # 0. Setup
    k_fold_cross_validator = KFold(10, random_state=random_seed, shuffle=True)

    # 1. Cross-validation training
    k_fold_scores = []
    for train_index, test_index in k_fold_cross_validator.split(data):
        # 1.1. Initialize model
        model = LGBMRegressor(n_jobs=n_jobs, random_state=random_seed)

        # 1.2. Train model
        model.fit(
            np.ascontiguousarray(data.iloc[train_index]),
            np.ravel(np.ascontiguousarray(y.iloc[train_index]).reshape(-1, 1)),
        )

        # 1.3. Calculate R2 score
        k_fold_scores.append(
            r2_score(
                np.ravel(np.ascontiguousarray(y.iloc[test_index]).reshape(-1, 1)),
                model.predict(np.ascontiguousarray((data.iloc[test_index]))),
            )
        )

    return np.mean(k_fold_scores)
