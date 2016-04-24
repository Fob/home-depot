import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from src.algos.utils import rmse


class_labels = {
    7: 3,
    6: 2.67,
    5: 2.33,
    4: 2,
    3: 1.67,
    2: 1.33,
    1: 1
}

def decode_class(class_label):
    return class_labels[class_label]


train_df = pd.read_csv('./dataset/all_good_classes_train.csv', index_col='id')
y_train = train_df['class_label'].values
y_true_rel = train_df['relevance'].values
X_train = train_df.drop(['relevance', 'class_label'], axis=1).values

score = []
for train, test in KFold(len(y_train), n_folds=5, shuffle=True, random_state=42):
    X_cv_train = X_train[train]
    X_cv_test = X_train[test]
    y_cv_train = y_train[train]
    y_cv_test = y_train[test]

    #clf = LogisticRegression()
    #clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    #clf = ExtraTreesClassifier(n_estimators=300, bootstrap=True, max_features=5, min_samples_split=6)
    clf = SVC()
    clf.fit(X_cv_train, y_cv_train)

    y_class_pred = clf.predict(X_cv_test)
    y_rel_pred = np.array([decode_class(x) for x in y_class_pred])
    y_true = y_true_rel[test]
    score.append(rmse(y_true, y_rel_pred))
    print 'iteration finished..'
print np.mean(score)