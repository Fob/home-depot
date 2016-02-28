import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from scipy.sparse import vstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

data = pd.read_csv('dataset/train.csv')


vectorizer = CountVectorizer()
search_term_decoded=vectorizer.fit_transform(data['search_term'])


productVectorizer = CountVectorizer()
product_decoded = productVectorizer.fit_transform([str(i) for i in data['product_uid']])

X = hstack([product_decoded, search_term_decoded])
y=data['relevance']
y=(y-1)/2
y[y>0.5] = 1
y[y<=0.5] =0

print X.shape
print y.shape

clf = LogisticRegression()
clf.fit(X,y)

print np.sqrt(np.sum(((clf.predict_proba(X)[:,1] * 2) + 1 - data['relevance'])**2)/len(y))

dataTest = pd.read_csv('dataset/test.csv')
test_product_decoded = productVectorizer.transform([str(i) for i in dataTest['product_uid']])
test_search_term_decoded = vectorizer.transform(dataTest['search_term'])

XT = hstack([test_product_decoded, test_search_term_decoded])

print XT.shape

YT=(clf.predict_proba(XT)[:,1] * 2) + 1

out = pd.DataFrame({'id':dataTest['id'], 'relevance':YT})

out.to_csv('log_regression.csv', index=None)

# cross validation


y=data['relevance']
y_normilized=(y-1)/2

cv = KFold(len(y),10,True)
scores=[]
threasholds = range(5,95,5)
for th in threasholds:
    th= th/100.0
    print th
    y_th=np.array(y_normilized)
    y_th[y_th>th]=1
    y_th[y_th<=th]=0

    sc = []
    for train, test in cv:
        X_train = X.tocsc()[train,:]
        X_test = X.tocsc()[test,:]
        y_train = y_th[train]
        y_test = y[test]
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        sc = np.append(sc,np.sqrt(np.sum(((clf.predict_proba(X_test)[:, 1] * 2) + 1 - y_test) ** 2) / len(y)))
    scores=np.append(scores,sc.mean())


print scores
