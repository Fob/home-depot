import numpy as np
import datetime
from sklearn.cross_validation import KFold
from src.algos.utils import rmse


def run_cv(clf, X, y, n_folds=5):
    kfold = KFold(len(y), n_folds=n_folds, shuffle=True)
    sc = []
    iter = 1
    for train, test in kfold:
        print 'Iter #%d start' % iter
        start_time = datetime.datetime.now()

        X_train = X.tocsc()[train,:]
        X_test = X.tocsc()[test,:]
        y_train = y[train]
        y_test = y[test]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        sc = np.append(sc, rmse(y_test, y_pred))
        print 'Iter #%d end, time = %s' % (iter, (datetime.datetime.now() - start_time))
        iter += 1
    return sc.mean()
