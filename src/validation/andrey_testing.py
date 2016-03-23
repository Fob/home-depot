import src.features.extract_count_features as ext_c_ft
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from src.algos.utils import RMSE
import numpy as np
import src.features.w2v as w2v
import pandas as pd


train_df, test_df = ext_c_ft.load_features1('features_size')

# load W2V model
w2v_wiki_model = w2v.read_w2v_model_from_text_file("/media/andrey/ARCHIVE/w2v_trained/vectors_wiki1B_skipgram_s500_w5_neg20_hs0_sam1e-4_iter5.txt")
# ADD Word2Vec Columns
train_data = pd.read_csv('./dataset/train_no_stem_no_sw.csv', index_col='id')

w2v_doc_vect_features_train = w2v.get_features_sum_of_vects_as_doc(w2v_wiki_model, train_data)

# Cross-Validation
scores = []
thresholds = np.arange(5,100,5)/100.
cv = KFold(n=len(train_df), n_folds=10, shuffle=True, random_state=23)
for threshold in [0.05]:

    final_train_df = train_df.copy()
    final_train_df.iloc[10]

    final_train_df[w2v_doc_vect_features_train.columns.values] = w2v_doc_vect_features_train

    #w2v_count_synonyms_features_train = w2v.get_features_count_synonyms(w2v_wiki_model, train_data, threshold)
    #final_train_df[w2v_count_synonyms_features_train.columns.values] = w2v_count_synonyms_features_train

    X_train = final_train_df.drop('relevance', axis=1).values
    y_train = final_train_df['relevance'].values

    clf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42, max_depth=10, max_features=4)
    score = cross_val_score(clf, X_train, y_train, scoring=RMSE, cv=cv)
    scores = np.append(scores, score.mean())
    print 'CV score = %f for %f threshold' % (score.mean(), threshold)



# TRAIN ALL
#----------------------------------------------------
final_train_df = train_df.copy()

#w2v_doc_vect_features_train = w2v.get_features_sum_of_vects_as_doc(w2v_wiki_model, train_data)
final_train_df[w2v_doc_vect_features_train.columns.values] = w2v_doc_vect_features_train

threshold = 0.05
w2v_count_synonyms_features_train = w2v.get_features_count_synonyms(w2v_wiki_model, train_data, threshold)
final_train_df[w2v_count_synonyms_features_train.columns.values] = w2v_count_synonyms_features_train

X_train = final_train_df.drop('relevance', axis=1).values
y_train = final_train_df['relevance'].values
clf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42, max_depth=10, max_features=4)
clf.fit(X_train, y_train)
np.sqrt((1.0/len(y_train))*np.sum((y_train-clf.predict(X_train))**2))
#----------------------------------------------------


# TEST OUTPUT
#----------------------------------------------------
test_data = pd.read_csv('./dataset/test_no_stem_no_sw.csv', index_col='id')
final_test_df = test_df.copy()

w2v_doc_vect_features_test = w2v.get_features_sum_of_vects_as_doc(w2v_wiki_model, test_data)
final_test_df[w2v_doc_vect_features_test.columns.values] = w2v_doc_vect_features_test
w2v_count_synonyms_features_test = w2v.get_features_count_synonyms(w2v_wiki_model, test_data, threshold)
final_test_df[w2v_count_synonyms_features_test.columns.values] = w2v_count_synonyms_features_test

X_test = final_test_df.values
y_test = clf.predict(X_test)
out = pd.DataFrame({'id': test_df.index, 'relevance': y_test})
out.to_csv('./result/w2v_wiki_test_docandcos_thr_0_05.csv', index=None)
#----------------------------------------------------
