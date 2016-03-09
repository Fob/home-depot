import numpy as np
from sklearn.linear_model import LinearRegression

from src.algos.utils import rmse
from src.features.features2strmatrix import load_features
from src.features.features2strmatrix import product2attrs

p_to_a = product2attrs()
X_train, y_train, X_test, id_train, id_test = load_features(p2a=p_to_a)

clf = LinearRegression()
clf.fit(X_train, y_train)

y_predicted = clf.predict(X_train)
y_predicted[y_predicted < 1] = 1
y_predicted[y_predicted > 3] = 3
print 'full train error', rmse(y_train, y_predicted)
# 0.511923394592
# 0.512527745792
# 0.512562155434
# 0.512698447827


cc = np.hstack((p_to_a.columns, 'syn_combo'))
print len(cc[np.sum(X_train, axis=0) == 0])

vv = np.sum(X_train, axis=0)
column_weight = np.vstack((cc, vv))

print list(column_weight)
