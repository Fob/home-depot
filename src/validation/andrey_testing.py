from matplotlib import pyplot as plt
import src.features.extract_count_features as ext_c_ft
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from src.algos.utils import RMSE
import numpy as np
import src.features.w2v as w2v
import pandas as pd


#train_df, test_df = ext_c_ft.load_features1('features_size')

train_df = pd.read_csv('./src/features/builded_from_loadfeztures1_train.csv', index_col='id')


# load W2V model
w2v_wiki_model = w2v.read_w2v_model_from_text_file("/media/andrey/ARCHIVE/w2v_trained/vectors_wiki1B_skipgram_s500_w5_neg20_hs0_sam1e-4_iter5.txt")
# load W2V model
w2v_descr_title_model = w2v.read_w2v_model_from_text_file("/home/andrey/Kaggle/home-depot/dataset/all_text_w2v_model")
# ADD Word2Vec Columns
train_data = pd.read_csv('./dataset/train_no_stem_no_sw.csv', index_col='id')

w2v_doc_vect_features_train = w2v.get_features_sum_of_vects_as_doc_fast_method(w2v_wiki_model, train_data)

w2v_title_descr_doc_vect_features_train = w2v.get_features_sum_of_vects_as_doc_fast_method(w2v_descr_title_model, train_data)
w2v_title_descr_doc_vect_features_train.columns = w2v_title_descr_doc_vect_features_train.columns.values+'_title_descr'

cnt_in_titledescr = w2v.get_count_of_words(train_data)

# Cross-Validation
scores = []
thresholds = [0]
#thresholds = np.arange(10,60,10)/100.
#thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]
cv = KFold(n=len(train_df), n_folds=5, shuffle=True, random_state=23)
for threshold in thresholds:

    final_train_df = train_df.copy()
    final_train_df.iloc[10]


    final_train_df[w2v_doc_vect_features_train.columns.values] = w2v_doc_vect_features_train

    final_train_df[w2v_title_descr_doc_vect_features_train.columns.values] = w2v_title_descr_doc_vect_features_train

    final_train_df[cnt_in_titledescr.columns.values] = cnt_in_titledescr

    #w2v_count_synonyms_features_train = w2v.get_features_count_synonyms(w2v_wiki_model, train_data, threshold)
    #final_train_df[w2v_count_synonyms_features_train.columns.values] = w2v_count_synonyms_features_train

    X_train = final_train_df.drop('relevance', axis=1).values
    y_train = final_train_df['relevance'].values

    clf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42, max_depth=10, max_features=4)
    score = cross_val_score(clf, X_train, y_train, scoring=RMSE, cv=cv)
    scores = np.append(scores, score.mean())
    print 'CV score = %f for %f threshold' % (score.mean(), threshold)

plt.plot(np.array(thresholds), scores)
plt.show()

out = pd.DataFrame({'scores': scores})
out.to_csv('./result/scores.csv', index=None)

# TRAIN ALL
#----------------------------------------------------
final_train_df = train_df.copy()

w2v_doc_vect_features_train = w2v.get_features_sum_of_vects_as_doc_fast_method(w2v_wiki_model, train_data)
final_train_df[w2v_doc_vect_features_train.columns.values] = w2v_doc_vect_features_train

w2v_title_descr_doc_vect_features_train = w2v.get_features_sum_of_vects_as_doc_fast_method(w2v_descr_title_model, train_data)
w2v_title_descr_doc_vect_features_train.columns = w2v_title_descr_doc_vect_features_train.columns.values+'_title_descr'
final_train_df[w2v_title_descr_doc_vect_features_train.columns.values] = w2v_title_descr_doc_vect_features_train

cnt_in_titledescr = w2v.get_count_of_words(train_data)
final_train_df[cnt_in_titledescr.columns.values] = cnt_in_titledescr

#threshold = 0.04
#w2v_count_synonyms_features_train = w2v.get_features_count_synonyms(w2v_wiki_model, train_data, threshold)
#final_train_df[w2v_count_synonyms_features_train.columns.values] = w2v_count_synonyms_features_train

X_train = final_train_df.drop('relevance', axis=1).values
y_train = final_train_df['relevance'].values
clf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42, max_depth=10, max_features=4)
clf.fit(X_train, y_train)
np.sqrt((1.0/len(y_train))*np.sum((y_train-clf.predict(X_train))**2))
out = final_train_df[['sim_with_title_w2v', 'sim_with_descr_w2v', 'sim_with_title_w2v_title_descr', 'sim_with_descr_w2v_title_descr', 'cnt_in_title', 'cnt_in_descr']]

out.to_csv('./src/features/w2v_wiki_and_descr_without_typos_train_features.csv', index=None)
#----------------------------------------------------


# TEST OUTPUT
#----------------------------------------------------
test_df = pd.read_csv('./src/features/builded_from_loadfeztures1_test.csv', index_col='id')
test_data = pd.read_csv('./dataset/test_no_stem_no_sw.csv', index_col='id')
final_test_df = test_df.copy()

#on w2v wiki
w2v_doc_vect_features_test = w2v.get_features_sum_of_vects_as_doc_fast_method(w2v_wiki_model, test_data)
final_test_df[w2v_doc_vect_features_test.columns.values] = w2v_doc_vect_features_test
#on w2v title_descr
w2v_title_descr_doc_vect_features_test = w2v.get_features_sum_of_vects_as_doc_fast_method(w2v_descr_title_model, test_data)
w2v_title_descr_doc_vect_features_test.columns = w2v_title_descr_doc_vect_features_test.columns.values+'_title_descr'
final_test_df[w2v_title_descr_doc_vect_features_test.columns.values] = w2v_title_descr_doc_vect_features_test
#just count of words
cnt_in_titledescr_test = w2v.get_count_of_words(test_data)
final_test_df[cnt_in_titledescr_test.columns.values] = cnt_in_titledescr_test
#synonyms
#w2v_count_synonyms_features_test = w2v.get_features_count_synonyms(w2v_wiki_model, test_data, threshold)
#final_test_df[w2v_count_synonyms_features_test.columns.values] = w2v_count_synonyms_features_test


X_test = final_test_df.values
y_test = clf.predict(X_test)
out = pd.DataFrame({'id': test_df.index, 'relevance': y_test})
out.to_csv('./result/w2v_wiki_and_descr.csv', index=None)
#----------------------------------------------------




#----------------------------------------------------
#TEST NEW FUNCTIONALITY
#----------------------------------------------------
train_data = pd.read_csv('./dataset/train_no_stem_no_sw.csv', index_col='id')


with open('./dataset/titles.txt', 'a') as f:
    train_data['product_title'].to_csv(f, index=None, header=False)
#train_data['product_title'][1:10].to_csv('./dataset/all_text.txt', index=None)


w2v_descr_model = w2v.read_w2v_model_from_text_file("/home/andrey/word2vec/home-depot/vectors")

word='makita'
w2v.cos_sim(w2v_wiki_model[word], w2v_descr_model[word])

