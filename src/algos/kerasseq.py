import logging
import os.path
import sys

from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential

from src.algos.utils import rmse
from src.features.features2strmatrix import features_to_x
from src.features.features2strmatrix import match_features
from src.features.features2strmatrix import zero_normalization
import pandas as pd

# Logging
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)


# Main


def keras_calc(X_train, y_train, X_test):
    model = Sequential()
    # 2 inputs, 10 neurons in 1 hidden layer, with tanh activation and dropout
    model.add(Dense(30, init='uniform', input_dim=38))
    model.add(Activation('tanh'))
    model.add(Dropout(0.25))

    model.add(Dense(30, init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))


    # 1 output, linear activation
    model.add(Dense(1, init='uniform'))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')

    model.fit(X_train, y_train, nb_epoch=50, batch_size=32)
    y_predicted = model.predict(X_train, 32)

    y_predicted[y_predicted < 1] = 1
    y_predicted[y_predicted > 3] = 3
    print 'train error', rmse(y_train, y_predicted)

    y_test = model.predict(X_test, 32)
    y_test[y_test < 1] = 1
    y_test[y_test > 3] = 3

    return y_test[:, 0]


# features = match_features()
# features_normalized = zero_normalization(features, merge=False)
# features_validated = select_features('rf.2', features_normalized)
train_df = pd.read_csv('./dataset/good_ft_2_train.csv', index_col='id')
# features = match_features(data_file='test')
# features_normalized = zero_normalization(features, merge=False)
# features_validated = select_features('rf.2', features_normalized)
test_df = pd.read_csv('./dataset/good_ft_2_test.csv', index_col='id')
y_train = train_df['relevance']
X_train = features_to_x(train_df)
X_test = features_to_x(test_df)

y_test = keras_calc(X_train.as_matrix(), y_train, X_test.as_matrix())

out = pd.DataFrame({'id': X_test.index, 'relevance': y_test})
out.to_csv('./result/keras.csv', index=None)
print 'saved ./result/keras.csv'
