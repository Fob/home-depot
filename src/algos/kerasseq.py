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
features = match_features()
features_normalized = zero_normalization(features, merge=False)
# features_validated = select_features('rf.2', features_normalized)
features_validated = features_normalized

y_train = features_validated['relevance']
X_train = features_to_x(features_validated)

model = Sequential()
# 2 inputs, 10 neurons in 1 hidden layer, with tanh activation and dropout
model.add(Dense(10, init='uniform', input_dim=417))
model.add(Activation('tanh'))
model.add(Dropout(0.25))
# 1 output, linear activation
model.add(Dense(1, init='uniform'))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer='rmsprop')

model.fit(X_train.as_matrix(), y_train, nb_epoch=5, batch_size=32)
y_predicted = model.predict(X_train.as_matrix(), 32)

y_predicted[y_predicted < 1] = 1
y_predicted[y_predicted > 3] = 3
print 'train error', rmse(y_train, y_predicted)

features = match_features(data_file='test')
features2 = match_features(data_file='train')
features_normalized = zero_normalization(features, merge=False)
features_normalized2 = zero_normalization(features2, merge=False)
# features_validated = select_features('rf.2', features_normalized)
features_validated = features_normalized

X_test = features_to_x(features_validated)

y_test = model.predict(X_test.as_matrix(), 32)
y_test[y_test < 1] = 1
y_test[y_test > 3] = 3

out = pd.DataFrame({'id': features_validated['id'], 'relevance': y_test[:,0]})
out.to_csv('./result/keras.csv', index=None)
print 'saved ./result/keras.csv'