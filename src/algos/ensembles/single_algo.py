import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from src.algos.ensembles.single_model_cv import train_clf


train_df = pd.read_csv('./dataset/all_good_features_train.csv', index_col='id')
y_train = train_df['relevance'].values
X_train = train_df.drop('relevance', axis=1).values


# RandomForestRegressor (n_estimators=1000, max_depth=10, max_features=4)
train_clf(
    {'n_estimators': [100], 'max_depth': [10], 'max_features': [4]},
    RandomForestRegressor(),
    X_train, y_train
)

# ExtraTreesRegressor
train_clf(
    {'n_estimators': [300], 'bootstrap': [True], 'max_features': [5,6,8,12,17], 'min_samples_split': [2,4,7,10]}, #'max_features': [4], min_samples_split
    ExtraTreesRegressor(),
    X_train, y_train
)
