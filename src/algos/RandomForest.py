import logging
import os.path
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.algos.utils import rmse
from src.features.features2strmatrix import features_to_x
from src.features.features2strmatrix import match_features
from src.features.features2strmatrix import normalize_search_len
from src.features.features2strmatrix import select_features
from src.features.features2strmatrix import zero_normalization

# Logging
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
# Main
features = match_features()
features_normalized = zero_normalization(features, merge=False)
features_validated = select_features('rf.2', features_normalized)
search_len_normalized = normalize_search_len(features_validated)
search_len_normalized = select_features('rf.slennorm.2', search_len_normalized)

y_train = search_len_normalized['relevance']
X_train = features_to_x(search_len_normalized)

clf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=257)
clf.fit(X_train, y_train)

y_predicted = clf.predict(X_train)
y_predicted[y_predicted < 1] = 1
y_predicted[y_predicted > 3] = 3
print 'train error', rmse(y_train, y_predicted)

features = match_features(data_file='test')
features_normalized = zero_normalization(features, merge=False)
features_validated = select_features('rf.2', features_normalized)
search_len_normalized = normalize_search_len(features_validated)
search_len_normalized = select_features('rf.slennorm.2', search_len_normalized)

X_test = features_to_x(search_len_normalized)

y_test = clf.predict(X_test)
y_test[y_test < 1] = 1
y_test[y_test > 3] = 3

out = pd.DataFrame({'id': search_len_normalized['id'], 'relevance': y_test})
out.to_csv('./result/random_forest_regression.csv', index=None)
print 'saved ./result/random_forest_regression.csv'
# 0.52430
