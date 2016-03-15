from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


RMSE = make_scorer(rmse, greater_is_better=False)


def rmse_normalized(y_true, y_pred):
    y_pred[y_pred < 1] = 1
    y_pred[y_pred > 3] = 3
    return mean_squared_error(y_true, y_pred) ** 0.5


RMSE_NORMALIZED = make_scorer(rmse_normalized, greater_is_better=False)
