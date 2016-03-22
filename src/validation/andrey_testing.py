import src.features.extract_count_features as ext_c_ft
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from src.algos.utils import RMSE
import numpy as np
import src.features.w2v as w2v
import pandas as pd


train_df, test_df = ext_c_ft.load_features1()

# load W2V model
w2v_wiki_model = w2v.read_w2v_model_from_text_file("/media/andrey/ARCHIVE/w2v_trained/vectors_wiki1B_skipgram_s500_w5_neg20_hs0_sam1e-4_iter5.txt")
# ADD Word2Vec Columns
train_data = pd.read_csv('./dataset/train_features_size.csv', index_col='id')

# Cross-Validation
scores = []
thresholds = np.arange(5,100,5)/100.
cv = KFold(n=len(X_train), n_folds=10, shuffle=True, random_state=23)
for threshold in [1]:

    w2v_dov_vect_features_train = w2v.get_features_sum_of_vects_as_doc(w2v_wiki_model, train_data)
    w2v_count_synonyms_features_train = w2v.get_features_count_synonyms(w2v_wiki_model, train_data, threshold)
    train_df[w2v_dov_vect_features_train.columns.values] = w2v_dov_vect_features_train
    train_df[w2v_count_synonyms_features_train.columns.values] = w2v_count_synonyms_features_train
    train_df.iloc[0]

    final_train_df = train_df
    final_train_df.iloc[10]
    X_train = final_train_df.ix[:, final_train_df.columns != 'relevance'].values
    y_train = train_df['relevance'].values

    clf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42, max_depth=10, max_features=4)
    score = cross_val_score(clf, X_train, y_train, scoring=RMSE, cv=cv)
    scores = np.append(scores, score.mean())
    print 'CV score = %f for %f threshold' % (score.mean(), threshold)



# TRAIN ALL
#w2v_features_train = w2v.get_features_count_synonyms(w2v_wiki_model, train_data, 0.1)
#train_df[w2v_features_train.columns.values] = w2v_features_train
final_train_df = train_df
X_train = final_train_df.ix[:, final_train_df.columns != 'relevance'].values
y_train = train_df['relevance'].values
clf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42, max_depth=10, max_features=4)
clf.fit(X_train, y_train)
np.sqrt((1.0/len(y_train))*np.sum((y_train-clf.predict(X_train))**2))


# TEST OUTPUT
test_data = pd.read_csv('./dataset/test_features_size.csv', index_col='id')

w2v_dov_vect_features_test = w2v.get_features_sum_of_vects_as_doc(w2v_wiki_model, test_data)
w2v_count_synonyms_features_test = w2v.get_features_count_synonyms(w2v_wiki_model, test_data, 0.1)
test_df[w2v_dov_vect_features_test.columns.values] = w2v_dov_vect_features_test
test_df[w2v_count_synonyms_features_test.columns.values] = w2v_count_synonyms_features_test


final_test_df = test_df
X_test = final_test_df.values
y_test = clf.predict(X_test)
out = pd.DataFrame({'id': test_df.index, 'relevance': y_test})
out.to_csv('./result/w2v_wiki_test_syn_and_doc_0_1.csv', index=None)
