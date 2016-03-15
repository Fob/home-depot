import logging
import os.path
import sys

import sklearn.linear_model as ln
from sklearn import cross_validation
from sklearn.cross_validation import KFold

from src.algos.utils import RMSE_NORMALIZED
from src.algos.utils import rmse
from src.features.features2strmatrix import features_to_x
from src.features.features2strmatrix import match_features
from src.features.features2strmatrix import zero_normalization

# Logging
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
# Main
features = match_features()
features_normalized = zero_normalization(features)
# test = prepare_word_set('test')

y_train = features['relevance']

X_train = features_to_x(features_normalized)

a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
# alphas=[0.1 * x for x in a],
clf = ln.RidgeCV(alphas=[0.001 * x for x in a], normalize=True)
clf.fit(X_train, y_train)

y_predicted = clf.predict(X=X_train)
y_predicted[y_predicted < 1] = 1
y_predicted[y_predicted > 3] = 3
print 'full train error', rmse(y_train, y_predicted)
# 0.511923394592
# 0.512527745792
# 0.512562155434
# 0.512698447827
# 0.496877387728

cv = KFold(len(y_train), n_folds=5)
print 'cross validation score', cross_validation.cross_val_score(
    ln.Ridge(alpha=0.007, normalize=True), X_train, y_train, scoring=RMSE_NORMALIZED, cv=cv).mean()
# cc = np.hstack((p_to_a.columns, 'syn_combo'))
# print len(cc[np.sum(X_train, axis=0) == 0])
# -0.502024170078
# -0.502024170078
# -0.501908125014
