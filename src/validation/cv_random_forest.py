import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import src.features.extract_features as ext_ft
from src.validation.cv import run_cv

test_data = pd.read_csv('./dataset/test.csv')

for vect in ['cnt', 'tfidf']:
    X_train, y_train, X_test = ext_ft.get_all_in_one_feature_matrix(vect=vect)

    print 'X_train.shape = %s; y_train.shape = %s; X_test.shape = %s' % (X_train.shape, y_train.shape, X_test.shape)

    #X_train_red = X_train.tocsc()[0:5000,:]
    #y_train_red = y_train[0:5000]
    #X_test_red = X_test.tocsc()[0:1000,:]

    clf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=257)
    clf.fit(X_train, y_train)
    #clf.fit(X_train_red, y_train_red)

    pred = clf.predict(X_test)
    #pred = clf.predict(X_test_red)

    print 'prediction done for %s' % vect

    #test_data = test_data[0:1000]

    out = pd.DataFrame({'id': test_data['id'], 'relevance': pred})
    out.to_csv('./result/rf1_%s.csv' % vect, index=None)


'''
print '1'
clf = RandomForestRegressor()
clf.fit(X.tocsc()[1:5000,:], y[1:5000])
print '2'
pred = clf.predict(X.tocsc()[1:5000,:])
print 'RMSE=%f' % rmse(y[1:5000], pred)

clf = RandomForestRegressor()
score = run_cv(clf, X.tocsc()[0:5000,:], y[0:5000])
score = run_cv(clf, X, y)
'''