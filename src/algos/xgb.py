import pandas as pd
import numpy as np
import xgboost as xgb
import src.features.extract_count_features as ext_c_ft
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from src.algos.utils import RMSE
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from src.algos.utils import rmse


train_df, test_df = ext_c_ft.load_features1()

y_train = train_df['relevance'].values
X_train = train_df.drop('relevance', axis=1).values
X_test = test_df.ix[:, :'ratio_material'].values

xgbr = xgb.XGBRegressor(max_depth=4, n_estimators=200)
xgbr.fit(X_train, y_train)
xgb_pred = xgbr.predict(X_test)
print 'XGBoost finished'

rfr = RandomForestRegressor(n_estimators=1000, n_jobs=-1, max_depth=10, max_features=4)
rfr.fit(X_train, y_train)
rfr_pred = rfr.predict(X_test)
print 'RFR finished'

lin_reg = Ridge(alpha=0.001, normalize=True)
lin_reg.fit(X_train, y_train)
lin_reg_pred = lin_reg.predict(X_test)
lin_reg_pred[lin_reg_pred < 1] = 1
lin_reg_pred[lin_reg_pred > 3] = 3
print 'Linear Regression finished'

result_pred = (xgb_pred + rfr_pred + lin_reg_pred)/3

out = pd.DataFrame({'id': test_df.index, 'relevance': result_pred})
out.to_csv('./result/rf_xgb_rfr_linreg.csv', index=None)

'''
cv = KFold(len(y_train), n_folds=5, shuffle=True, random_state=42)
for alpha in np.logspace(-5, 5, 11):
    sc = []
    for train, test in cv:
        X1 = X_train[train]
        X2 = X_train[test]
        y1 = y_train[train]
        y2 = y_train[test]

        clf = Ridge(alpha=alpha, normalize=True)
        clf.fit(X1, y1)

        y_pred = clf.predict(X2)
        y_pred[y_pred < 1] = 1
        y_pred[y_pred > 3] = 3
        sc.append(rmse(y2, y_pred))

    print 'For alpha = %f score = %f' % (alpha, np.array(sc).mean())

cv = KFold(len(y_train), n_folds=5, shuffle=True)
clf = xgb.XGBRegressor(max_depth=4, n_estimators=200, learning_rate=0.25)
score = cross_val_score(clf, X_train, y_train, scoring=RMSE, cv=cv)
print 'CV score = %f' % score.mean()

params = {'max_depth': np.arange(4)+1, 'n_estimators': [100, 200, 300]}
xgbr = xgb.XGBRegressor()
cv = KFold(len(y_train), n_folds=5, shuffle=True, random_state=23)
clf = GridSearchCV(xgbr, params, scoring=RMSE, n_jobs=-1, cv=cv)
clf.fit(X_train, y_train)
print 'Best params = ', clf.best_params_
print 'Best score = ', clf.best_score_

xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True,
                 objective="reg:linear", nthread=-1, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)
'''