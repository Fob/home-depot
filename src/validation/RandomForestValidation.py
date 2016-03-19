import logging
import os.path
import sys

from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor

from src.algos.utils import RMSE_NORMALIZED
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

# select raw features
select_features('rf', features_normalized, allow_merge=False,
                cls=RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42))
select_features('rf.1', features_normalized, allow_merge=False,
                cls=RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
                start_mask='rf')
features_validated = select_features('rf.2', features_normalized, allow_merge=True,
                                     cls=RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
                                     start_mask='rf.1')

search_len_normalized = normalize_search_len(features_validated)
select_features('rf.slennorm', search_len_normalized, allow_merge=False,
                cls=RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
                start_mask='rf.2')
select_features('rf.slennorm.1', search_len_normalized, allow_merge=False,
                cls=RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
                start_mask='rf.slennorm')
search_len_normalized = select_features('rf.slennorm.2', search_len_normalized, allow_merge=False,
                                        cls=RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
                                        start_mask='rf.slennorm.1')

y_train = search_len_normalized['relevance']
X_train = features_to_x(search_len_normalized)

print 'cross validation score', cross_validation.cross_val_score(
    RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42), X_train, y_train, scoring=RMSE_NORMALIZED,
    cv=5).mean()


# -0.490554695743
# -0.490441847191
# -0.49004546468
# -0.483139294024
