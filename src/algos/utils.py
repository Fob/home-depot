from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

RMSE = make_scorer(rmse, greater_is_better=False)