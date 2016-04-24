import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from src.algos.ensembles.single_model_cv import train_clf
import datetime


train_df = pd.read_csv('./dataset/good_ft_3_train.csv', index_col='id')
y_train = train_df['relevance'].values
X_train = train_df.drop(['relevance'], axis=1).values

test_df = pd.read_csv('./dataset/good_ft_3_test.csv', index_col='id')
X_test = test_df.values


''' START '''
X_train_with_meta = X_train.copy()
X_test_with_meta = X_test.copy()
clfs = [
    RandomForestRegressor(n_estimators=1000, max_depth=10, max_features=4, n_jobs=-1),
    Ridge(alpha=1),
    Lasso(alpha=1e-06, max_iter=1e5),
    #xgb.XGBRegressor(max_depth=4, n_estimators=200),
    KNeighborsRegressor(n_neighbors=1000, n_jobs=-1),
    ExtraTreesRegressor(n_estimators=100, bootstrap=True, max_features=5, min_samples_split=6, n_jobs=-1)
]
N = 3
K = 5
start_time = datetime.datetime.now()
for clf in clfs:
    meta_x = np.zeros(len(y_train))
    meta0_x = np.zeros(len(X_test))
    for n in range(0,N):
        meta_xn = np.zeros(len(y_train))
        meta0_xn = np.zeros(len(X_test))

        for train, test in KFold(len(y_train), n_folds=K, shuffle=True):
            X_cv_train = X_train[train]
            X_cv_test = X_train[test]
            y_cv_train = y_train[train]
            y_cv_test = y_train[test]

            clf.fit(X_cv_train, y_cv_train)

            meta_xnk = clf.predict(X_cv_test)
            meta_xnk[meta_xnk < 1] = 1
            meta_xnk[meta_xnk > 3] = 3

            meta_xn[test] = meta_xnk
            meta0_xnk = clf.predict(X_test)
            meta0_xn = meta0_xn + meta0_xnk

        meta0_xn /= K
        meta0_x = meta0_x + meta0_xn
        meta_x = meta_x + meta_xn

    meta0_x /= N
    meta_x /= N
    X_test_with_meta = np.hstack((X_test_with_meta, meta0_x.reshape(len(X_test),1)))
    X_train_with_meta = np.hstack((X_train_with_meta, meta_x.reshape(len(y_train),1)))
    print 'Clf finished:', clf
end_time = datetime.datetime.now()
print 'Done. Time = ', (end_time - start_time)


xgbr = xgb.XGBRegressor(max_depth=4, n_estimators=200)
xgbr.fit(X_train_with_meta, y_train)
print 'Done'
pred = xgbr.predict(X_test_with_meta)

out = pd.DataFrame({'id': test_df.index, 'relevance': pred})
out.to_csv('./result/ensemble_tfidf.csv', index=None)















#X_train_with_meta = np.hstack((X_train, meta_x.reshape(len(y_train),1)))

et = ExtraTreesRegressor(n_estimators=300, bootstrap=True, max_features=5, min_samples_split=6)
et.fit(X_train_with_meta, y_train)
print 'Done'
pred = et.predict(X_test_with_meta)

out = pd.DataFrame({'id': test_df.index, 'relevance': pred})
out.to_csv('./result/ensemble_tfidf_no_avg.csv', index=None)


# No ensemble
et = ExtraTreesRegressor(n_estimators=300, bootstrap=True, max_features=5, min_samples_split=6)
et.fit(X_train, y_train)
print 'Done'
pred = et.predict(X_test)

out = pd.DataFrame({'id': test_df.index, 'relevance': pred})
out.to_csv('./result/et_tfidf_no_avg.csv', index=None)


train_clf(
    {'n_estimators': [1000], 'max_depth': [10], 'max_features': [4]},
    RandomForestRegressor(n_jobs=-1, random_state=42),
    X_train_with_meta, y_train
)

train_clf(
    {'n_estimators': [300], 'bootstrap': [True], 'max_features': [5], 'min_samples_split': [6]},
    ExtraTreesRegressor(n_jobs=-1, random_state=42),
    X_train_with_meta, y_train
)

train_clf(
    {'n_estimators': [200], 'max_depth': [4]},
    xgb.XGBRegressor(),
    X_train_with_meta, y_train
)