import sklearn.linear_model as ln
from sklearn import cross_validation
from sklearn.cross_validation import KFold

from src.algos.utils import RMSE
from src.algos.utils import rmse
from src.features.features2strmatrix import load_features
from src.features.features2strmatrix import product2attrs

p_to_a = product2attrs()
X_train, y_train, X_test, id_train, id_test = load_features(p2a=p_to_a, merge_factor=0)

a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
# alphas=[0.1 * x for x in a],
clf = ln.RidgeCV(alphas=[0.001 * x for x in a], normalize=True)
clf.fit(X_train, y_train)

y_predicted = clf.predict(X_train)
y_predicted[y_predicted < 1] = 1
y_predicted[y_predicted > 3] = 3
print 'full train error', rmse(y_train, y_predicted)
# 0.511923394592
# 0.512527745792
# 0.512562155434
# 0.512698447827


cv = KFold(len(y_train), n_folds=5)
print 'cross validation score', cross_validation.cross_val_score(
    ln.Ridge(alpha=0.007, normalize=True), X_train, y_train, scoring=RMSE, cv=cv)

# cc = np.hstack((p_to_a.columns, 'syn_combo'))
# print len(cc[np.sum(X_train, axis=0) == 0])