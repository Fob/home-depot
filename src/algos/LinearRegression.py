import pandas as pd
from sklearn.linear_model import LinearRegression

from src.algos.utils import rmse
from src.features.features2strmatrix import load_features

X_train, y_train, X_test, id_train, id_test = load_features()

clf = LinearRegression()
clf.fit(X_train, y_train)

y_predicted = clf.predict(X_train)
y_predicted[y_predicted < 1] = 1
y_predicted[y_predicted > 3] = 3
print 'train error', rmse(y_train, y_predicted)

y_test = clf.predict(X_test)
y_test[y_test < 1] = 1
y_test[y_test > 3] = 3

out = pd.DataFrame({'id': id_test, 'relevance': y_test})
out.to_csv('./result/linear_regression.csv', index=None)
