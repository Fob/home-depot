import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import src.features.extract_features as ext_ft

test_data = pd.read_csv('./dataset/test.csv')

for vect in ['cnt', 'tfidf']:
    X_train, y_train, X_test = ext_ft.get_all_in_one_feature_matrix(vect=vect)

    print 'X_train.shape = %s; y_train.shape = %s; X_test.shape = %s' % (X_train.shape, y_train.shape, X_test.shape)

    clf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=257)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    print 'prediction done for %s' % vect

    out = pd.DataFrame({'id': test_data['id'], 'relevance': pred})
    out.to_csv('./result/rf1_%s.csv' % vect, index=None)

'''




'''

import src.features.extract_count_features as ext_c_ft
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from src.algos.utils import RMSE

#ext_c_ft.extract_process_and_save_features1()

train_df, test_df = ext_c_ft.load_features1()

# ADD Word2Vec Columns
import src.features.get_count_features_by_word2vec as w2v
w2v_features_train = w2v.get_count_features_by_word2vec('./dataset/train_new1.csv', 0.7)
#w2v_features_train.iloc[0]
train_df[['ratio_in_descr_w2v', 'ratio_in_title_w2v', 'words_in_descr_w2v', 'words_in_title_w2v']] = w2v_features_train
#train_df.iloc[0]


#X_train = train_df.ix[:, :'ratio_material'].values
X_train = train_df.ix[:, train_df.columns != 'relevance'].values
y_train = train_df['relevance'].values
X_test = test_df.ix[:, :'ratio_material'].values


cv = KFold(len(y_train), n_folds=5, shuffle=True, random_state=23)
for n_tree in [100]:
    clf = RandomForestRegressor(n_estimators=n_tree, n_jobs=-1, random_state=42)
    score = cross_val_score(clf, X_train, y_train, scoring=RMSE, cv=cv)
    print 'CV score = %f for %d trees' % (score.mean(), n_tree)


clf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)
y_test = clf.predict(X_test)

out = pd.DataFrame({'id': test_df.index, 'relevance': y_test})
out.to_csv('./result/rf_counts_001.csv', index=None)

