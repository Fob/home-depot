import pandas as pd
from sklearn import linear_model

from src.algos.utils import rmse
from src.features.features2strmatrix import load_features

X_train, y_train, X_test, id_train, id_test = load_features()

clf = linear_model.Ridge(alpha=0.009, normalize=True)
# 0.02484
clf.fit(X_train, y_train)

y_predicted = clf.predict(X_train)
y_predicted[y_predicted < 1] = 1
y_predicted[y_predicted > 3] = 3
print 'train error', rmse(y_train, y_predicted)

y_test = clf.predict(X_test)
y_test[y_test < 1] = 1
y_test[y_test > 3] = 3

out = pd.DataFrame({'id': id_test, 'relevance': y_test})
out.to_csv('./result/linear_regression_regularized.csv', index=None)
# 0.52233
# 0.51889
# 0.51883
# 0.50061
