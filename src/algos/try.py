import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
from scipy.sparse import hstack
from scipy.sparse import vstack

data = pd.read_csv('dataset/train.csv')

data.head()

print data['name'].unique();

print [ s.split(' ') for s in data['search_term']]

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
search_term_decoded=vectorizer.fit_transform(data['search_term'])


productVectorizer = CountVectorizer()
product_decoded = productVectorizer.fit_transform([str(i) for i in data['product_uid']])


y=data['relevance']



X = hstack([product_decoded, search_term_decoded])

print y.shape
print X.shape

clf = LinearRegression()
clf.fit(X,y)

print np.sum((clf.predict(X)-y)**2) / len(y)

# testing

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

cv = KFold(len(y),5,True)
score = cross_val_score(clf,X,y,scoring='mean_squared_error', cv=cv)

print np.sqrt(-score.mean())

# test data

dataTest = pd.read_csv('dataset/test.csv')

test_product_decoded = productVectorizer.transform([str(i) for i in dataTest['product_uid']])
test_search_term_decoded = vectorizer.transform(dataTest['search_term'])

XT = hstack([test_product_decoded, test_search_term_decoded])

print XT.shape

YT = clf.predict(XT)

print YT.shape

print YT[:10]

Y_hacked = YT
Y_hacked[Y_hacked<1] = 1

Y_hacked[Y_hacked>3] = 3

out = pd.DataFrame({'id':dataTest['id'], 'relevance':YT})

out.to_csv('linear_regression.csv', index=None)
print Y_hacked.max()
print Y_hacked.min()

out = pd.DataFrame({'id':dataTest['id'], 'relevance':Y_hacked})

out.to_csv('linear_regression_hacked.csv', index=None)
