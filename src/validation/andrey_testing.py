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
import numpy as np
import src.features.get_count_features_by_word2vec as w2v
import src.features.w2v as w2v_all
import pandas as pd

#ext_c_ft.extract_process_and_save_features1()

train_df, test_df = ext_c_ft.load_features1()

# ADD Word2Vec Columns
w2v_features_train = w2v_all.get_features_cos_between_sum_of_vectors('./dataset/train_new1.csv')
train_df[['sim_with_title_w2v', 'sim_with_descr_w2v']] = w2v_features_train
train_df.iloc[0]

final_train_df = train_df
final_train_df.iloc[10]
X_train = final_train_df.ix[:, final_train_df.columns != 'relevance'].values
y_train = train_df['relevance'].values

clf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
cv = KFold(n=len(X_train), n_folds=10, shuffle=True, random_state=None)
score = cross_val_score(clf, X_train, y_train, scoring=RMSE, cv=cv)
print 'CV score = %f for %f threshold' % (score.mean(), 100)



# ALL TRAIN RESULT
clf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)
np.sqrt((1.0/len(y_train))*np.sum((y_train-clf.predict(X_train))**2))


# TEST OUTPUT
w2v_features_test = w2v_all.get_features_cos_between_sum_of_vectors('./dataset/test_new1.csv')
test_df[['sim_with_title_w2v', 'sim_with_descr_w2v']] = w2v_features_test

test_df.iloc[0]
final_test_df = test_df
final_test_df.iloc[0]
X_test = final_test_df.values
y_test = clf.predict(X_test)
out = pd.DataFrame({'id': test_df.index, 'relevance': y_test})
out.to_csv('./result/w2v_avg_vectors.csv', index=None)

