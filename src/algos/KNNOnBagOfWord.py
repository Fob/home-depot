import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from scipy.sparse import hstack
from sklearn.cross_validation import KFold

data = pd.read_csv('dataset/train.csv')



from sklearn.feature_extraction.text import HashingVectorizer
search_term_hv = HashingVectorizer(n_features=100)
search_term_decoded = search_term_hv.fit_transform(data['search_term'])
product_decoded_hv = HashingVectorizer(n_features=10)
product_decoded = product_decoded_hv.fit_transform([str(i) for i in data['product_uid']])

X = hstack([product_decoded, search_term_decoded])
y=data['relevance']

print X.shape
print y.shape

X = X.tocsc()[:50000,:]
y = y[:50000]

cv = KFold(len(y), 10, True)
scores=[]

for k in [1,2,5,10,20,30,50,100,500]:
    sc = []
    print k
    for train, test in cv:
        X_train = X.tocsc()[train,:]
        X_test = X.tocsc()[test,:]
        y_train = y[train]
        y_test = y[test]
        knn = KNeighborsRegressor(n_neighbors=k, algorithm="kd_tree")
        knn.fit(X_train, y_train)
        y_predict = knn.predict(X_test)
        sc = np.append(sc,np.sqrt(np.sum(((y_predict - y_test) ** 2) / len(y_test))))

    print sc.mean()
    scores = np.append(scores, sc.mean())



print scores

