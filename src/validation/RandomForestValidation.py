import logging
import os.path
import sys

from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor

from src.algos.utils import RMSE_NORMALIZED
from src.features.features2strmatrix import features_to_x
from src.features.features2strmatrix import match_features
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
cls = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42, max_depth=10, max_features=4)
select_features('rf', features_normalized, allow_merge=False, cls=cls, start_mask=[u'words_in_title', u'words_in_descr',
                                                                                   u'words_in_brand',
                                                                                   u'words_in_color',
                                                                                   u'words_in_material',
                                                                                   u'words_in_size',
                                                                                   u'words_in_weight', u'words_in_volt',
                                                                                   u'words_in_watt',
                                                                                   u'whole_query_in_title',
                                                                                   u'whole_query_in_descr',
                                                                                   u'brand_in_query',
                                                                                   u'brand_in_title', u'color_in_query',
                                                                                   u'color_in_title',
                                                                                   u'mat_in_query', u'number_in_query',
                                                                                   u'query_len', u'title_len',
                                                                                   u'descr_len', u'ratio_title',
                                                                                   u'ratio_descr',
                                                                                   u'count_of_search_term_bigram',
                                                                                   u'count_of_search_term_trigram',
                                                                                   u'count_of_search_term_biterm',
                                                                                   u'count_of_search_term_bigram_in_product_title',
                                                                                   u'count_of_search_term_biterm_in_product_title',
                                                                                   u'relevance', u'id.1',
                                                                                   u'sim_with_title_w2v',
                                                                                   u'sim_with_descr_w2v',
                                                                                   u'sim_with_title_w2v_title_descr',
                                                                                   u'sim_with_descr_w2v_title_descr',
                                                                                   u'search_title_tfidf_sum',
                                                                                   u'search_title_tfidf_min',
                                                                                   u'search_title_tfidf_max',
                                                                                   u'search_descr_tfidf_sum',
                                                                                   u'search_descr_tfidf_min',
                                                                                   u'search_descr_tfidf_max'])
select_features('rf.1', features_normalized, cls=cls, start_mask='rf')
features_validated = select_features('rf.2', features_normalized, cls=cls, start_mask='rf.1')

y_train = features_validated['relevance']
X_train = features_to_x(features_validated)

print 'cross validation score', cross_validation.cross_val_score(
    RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42, max_depth=10, max_features=4)
    , X_train, y_train, scoring=RMSE_NORMALIZED, cv=5).mean()


# -0.490554695743
# -0.490441847191
# -0.49004546468
# -0.483139294024
# -0.482038088824
