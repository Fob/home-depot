from matplotlib import pyplot as plt
#import src.features.extract_count_features as ext_c_ft
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from src.algos.utils import RMSE
import numpy as np
import src.features.w2v as w2v
import pandas as pd


#train_df, test_df = ext_c_ft.load_features1('no_stem_no_sw_spell_check')
#train_df.to_csv('./src/features/builded_cnt_features_from_train_no_stem_no_sw_spell_check.csv', header=True)
#test_df.to_csv('./src/features/builded_cnt_features_from_test_no_stem_no_sw_spell_check.csv', header=True)

#train_df = pd.read_csv('./src/features/builded_cnt_features_from_train_no_stem_no_sw_spell_check.csv', index_col='id')
train_df = pd.read_csv('./src/features/builded_from_loadfeztures1_train.csv', index_col='id')


# load W2V model
w2v_wiki_model = w2v.read_w2v_model_from_text_file("/media/andrey/ARCHIVE/w2v_trained/vectors_wiki1B_skipgram_s500_w5_neg20_hs0_sam1e-4_iter5.txt")
# load W2V model
w2v_descr_title_model = w2v.read_w2v_model_from_text_file("/home/andrey/Kaggle/home-depot/dataset/w2v_model_all_text_spell_check")
# ADD Word2Vec Columns
train_data = pd.read_csv('./dataset/train_no_stem_no_sw_spell_check.csv', index_col='id')

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

#plt.plot(np.array(thresholds), scores)
#plt.show()

#out = pd.DataFrame({'scores': scores})
#out.to_csv('./result/scores.csv', index=None)

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
clf = RandomForestRegressor(n_estimators=500, n_jobs=-1, max_depth=10, max_features=4)
clf.fit(X_train, y_train)
print np.sqrt((1.0/len(y_train))*np.sum((y_train-clf.predict(X_train))**2))

#w2v_features = final_train_df[['sim_with_title_w2v', 'sim_with_descr_w2v', 'sim_with_title_w2v_title_descr', 'sim_with_descr_w2v_title_descr']]
#w2v_features.to_csv('./src/features/w2v_features_train.csv')
#----------------------------------------------------


# TEST OUTPUT
#----------------------------------------------------
test_df = pd.read_csv('./src/features/builded_from_loadfeztures1_test.csv', index_col='id')
test_data = pd.read_csv('./dataset/test_no_stem_no_sw_spell_check.csv', index_col='id')
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

#w2v_features = final_test_df[['sim_with_title_w2v', 'sim_with_descr_w2v', 'sim_with_title_w2v_title_descr', 'sim_with_descr_w2v_title_descr']]
#w2v_features.to_csv('./src/features/w2v_features_test.csv')


X_test = final_test_df.values
y_test = clf.predict(X_test)
out = pd.DataFrame({'id': test_df.index, 'relevance': y_test})
out.to_csv('./result/w2v_wiki_and_descr_spell_check.csv', index=None)
#----------------------------------------------------


#----------------------------------------------------
#TEST TF-IDF
#----------------------------------------------------


train_data = pd.read_csv('./dataset/spell_check/train_spell_check.csv', index_col='id')
train_data.iloc[1]
train_descr = train_data[['descr']]
train_title = train_data[['product_title']]
train_search = train_data[['search_term']]

test_data = pd.read_csv('./dataset/spell_check/test_spell_check.csv', index_col='id')
test_data.iloc[1]
test_descr = test_data[['descr']]
test_title = test_data[['product_title']]
test_search = test_data[['search_term']]

all_descr_data = pd.concat([train_descr, test_descr], axis=0)
all_title_data = pd.concat([train_title, test_title], axis=0)


all_title_docs = list(all_title_data.values.reshape((1,len(all_title_data)))[0])
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_title = TfidfVectorizer()
vectorizer_title.fit(all_title_docs)

all_descr_docs = list(all_descr_data.values.reshape((1,len(all_descr_data)))[0])
vectorizer_descr = TfidfVectorizer()
vectorizer_descr.fit(all_descr_docs)

#train_data = pd.read_csv('./dataset/spell_check/train_spell_check.csv', index_col='id')
train_data = pd.read_csv('./dataset/spell_check/test_spell_check.csv', index_col='id')
search = train_data['search_term']
title = train_data['product_title']
descr = train_data['descr']

print "transform..."
vect_t_search = vectorizer_title.transform(search)
vect_title = vectorizer_title.transform(title)
vect_d_search = vectorizer_descr.transform(search)
vect_descr = vectorizer_descr.transform(descr)
print "end transform"

def sparse_max_row(csr_mat):
    ret = np.maximum.reduceat(csr_mat.data, csr_mat.indptr[:-1])
    ret[np.diff(csr_mat.indptr) == 0] = 0
    return ret
def sparse_min_row(csr_mat):
    ret = np.minimum.reduceat(csr_mat.data, csr_mat.indptr[:-1])
    ret[np.diff(csr_mat.indptr) == 0] = 0
    return ret


#title
matrix_search_title_tfidf = vect_t_search.multiply(vect_title).todense()
matrix_search_title_tfidf = vect_t_search.multiply(vect_title)


feature_s_title_tfidf_sum = np.array(matrix_search_title_tfidf.sum(axis=1).reshape(len(train_data)))[0]
#feature_s_title_tfidf_avg = np.array(matrix_search_title_tfidf.mean(axis=1).reshape(len(train_data)))[0]
feature_s_title_tfidf_max = sparse_max_row(matrix_search_title_tfidf)
feature_s_title_tfidf_min = sparse_min_row(matrix_search_title_tfidf)
print "feature created"
out = pd.DataFrame(feature_s_title_tfidf_sum)
out.columns = ['search_title_tfidf_sum']
out['search_title_tfidf_min'] = feature_s_title_tfidf_min
out['search_title_tfidf_max'] = feature_s_title_tfidf_max
#out['search_title_tfidf_avg'] = feature_s_title_tfidf_avg

#description

matrix_search_descr_tfidf = vect_d_search.multiply(vect_descr)

feature_s_descr_tfidf_sum = np.array(matrix_search_descr_tfidf.sum(axis=1).reshape(len(train_data)))[0]
#feature_s_title_tfidf_avg = np.array(matrix_search_title_tfidf.mean(axis=1).reshape(len(train_data)))[0]
feature_s_descr_tfidf_max = sparse_max_row(matrix_search_descr_tfidf)
feature_s_descr_tfidf_min = sparse_min_row(matrix_search_descr_tfidf)
print "feature created"
out['search_descr_tfidf_sum'] = feature_s_descr_tfidf_sum
out['search_descr_tfidf_min'] = feature_s_descr_tfidf_min
out['search_descr_tfidf_max'] = feature_s_descr_tfidf_max
#out['search_title_tfidf_avg'] = feature_s_title_tfidf_avg

out.to_csv('./src/features/tfidf_features_test.csv', index=None)
print "finished"

#----------------------------------------------------
#TEST NEW FUNCTIONALITY
#----------------------------------------------------
train_data = pd.read_csv('./dataset/train_no_stem_no_sw.csv', index_col='id')
test_data = pd.read_csv('./dataset/test_no_stem_no_sw.csv', index_col='id')


with open('./dataset/all_text_train_and_test.txt', 'a') as f:
    test_data['descr'].to_csv(f, index=None, header=False)
#train_data['product_title'][1:10].to_csv('./dataset/all_text.txt', index=None)


w2v_descr_model = w2v.read_w2v_model_from_text_file("/home/andrey/word2vec/home-depot/vectors")

word='makita'
w2v.cos_sim(w2v_wiki_model[word], w2v_descr_model[word])







import numpy as np
import random
N = 20
dimensions = [10, 5, 10]
data = np.random.randn(N, dimensions[0])   # each row will be a datum
labels = np.zeros((N, dimensions[2]))
for i in xrange(N):
    labels[i,random.randint(0,dimensions[2]-1)] = 1

params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )


i=2
m = data.shape[0]
a1 = np.hstack((data, np.zeros((m,1)))).transpose()
W1_with_b = np.vstack((W1, b1)).transpose()
z2 = np.dot(W1_with_b, a1)
a2 = sigmoid(z2)

a2 = np.vstack((a2, np.zeros((1,m))))
W2_with_b = np.vstack((W2, b2)).transpose()
z3 = np.dot(W2_with_b, a2)
a3 = softmax(z3.transpose())















