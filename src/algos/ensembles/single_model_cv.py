from sklearn.cross_validation import KFold
from src.algos.utils import RMSE_NORMALIZED
import datetime
from sklearn.grid_search import GridSearchCV


def train_clf(params, clf, X_train, y_train):
    start = datetime.datetime.now()

    cv = KFold(len(y_train), n_folds=5, shuffle=True, random_state=42)
    grid = GridSearchCV(clf, params, scoring=RMSE_NORMALIZED, n_jobs=-1, cv=cv)
    grid.fit(X_train, y_train)

    end = datetime.datetime.now()
    print 'Best params = ', grid.best_params_
    print 'Best score = ', grid.best_score_
    print 'Time = ', (end - start)