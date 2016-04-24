import pandas as pd
from src.algos.ensembles.single_model_cv import train_clf
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import xgboost as xgb


train_df = pd.read_csv('./dataset/all_good_features_train.csv', index_col='id')
y_train = train_df['relevance'].values
X_train = train_df.drop('relevance', axis=1).values


# RandomForestRegressor
# Best: n_estimators=1000, max_depth=10, max_features=4
train_clf(
    {'n_estimators': [1000], 'max_depth': [10], 'max_features': [4]},
    RandomForestRegressor(n_jobs=-1, random_state=42),
    X_train, y_train
)

# ExtraTreesRegressor
# Best: n_estimators=300, bootstrap=True, max_features=5, min_samples_split=6
train_clf(
    {'n_estimators': [300], 'bootstrap': [True], 'max_features': [5], 'min_samples_split': [6]},
    ExtraTreesRegressor(n_jobs=-1, random_state=42),
    X_train, y_train
)

# Sklearn GradientBoostingRegressor
# todo: tune
train_clf(
    {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3], 'max_features': ['auto'], 'min_samples_split': [2]},
    GradientBoostingRegressor(random_state=42),
    X_train, y_train
)

# SVR
# todo tune
train_clf(
    {'C': [1.0], 'epsilon': [0.1], 'kernel': ['rbf'], 'gamma': ['auto']},
    SVR(),
    X_train, y_train
)

# Ridge
# Best: alpha=1
train_clf(
    {'alpha': [1]},
    Ridge(random_state=42),
    X_train, y_train
)

# Lasso
# Best: alpha=1e-06, max_iter=1e5
train_clf(
    {'alpha': [1e-06], 'max_iter': [1e5]},
    Lasso(random_state=42),
    X_train, y_train
)

# XGBoost linear
# Best: max_depth=4, n_estimators=200
train_clf(
    {'max_depth': [4], 'n_estimators': [200]},
    xgb.XGBRegressor(),
    X_train, y_train
)
