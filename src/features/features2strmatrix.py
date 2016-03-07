import re

import numpy as np
import pandas as pd
from pandas import DataFrame

false_words = {'of', 'a', 'the', 'from', 'into', 'and', 'or', 'in', 'to', 'with', '&', 'x', 'this',
               '&amp;', 'in.', 'are'}


def column_transformer(x, combine_sizes=True):
    c = str(x).lower()
    c = re.sub('\s', '_', c)
    if combine_sizes:
        if re.search('(:?width|height|depth|length)', c) is not None:
            return 'combined_size'
    c = re.sub('bullet\d+', 'bullet', c)
    return c


def value_transformer(c, v, skip_false_words=True):
    v = str(v).lower()
    av = set(re.split('[\s,]', v))
    if skip_false_words:
        av = av - false_words
    return av


def product2attrs(combine_sizes=True, skip_false_words=True):
    attrs = pd.read_csv('./dataset/attributes.csv')
    attrs = attrs[attrs['product_uid'] == attrs['product_uid']]
    descrs = pd.read_csv('./dataset/product_descriptions.csv')
    descrs = descrs[descrs['product_uid'] == descrs['product_uid']]

    print 'attributes: ' + str(attrs.shape)
    colls = attrs['name'].apply(lambda c: column_transformer(c, combine_sizes)).unique()
    print 'attributes columns: ' + str(len(colls))

    product_ids = [int(x) for x in pd.concat([attrs['product_uid'], descrs['product_uid']]).unique()]
    print 'unique ids: ' + str(len(product_ids))

    rs = DataFrame(index=product_ids, columns=np.hstack((colls, 'full_descr')))

    for index, row in attrs.iterrows():
        if index % 100000 == 0: print 'processed: ' + str(index)
        cc = column_transformer(row['name'], combine_sizes)
        cv = value_transformer(cc, row['value'], skip_false_words)
        id = int(row['product_uid'])
        current = rs.at[id, cc]
        if type(current) is float:
            rs.at[id, cc] = cv
        else:
            rs.at[id, cc] = current | cv

    print 'descriptions :' + str(descrs.shape)

    for index, row in descrs.iterrows():
        if index % 10000 == 0: print 'processed descr: ' + str(index)
        id = int(row['product_uid'])
        if id not in rs.index: continue
        rs.at[id, 'full_descr'] = value_transformer('full_descr', row['product_description'], skip_false_words)

    print 'result:' + str(rs.shape)
    return rs


def count_words(data, search):
    if type(data) is float:
        return 0
    return len(data & search)


def load_features(combine_sizes=True, skip_false_words=True):
    p2a = product2attrs(combine_sizes, skip_false_words)
    train_data = pd.read_csv('./dataset/train.csv')
    y_train = train_data['relevance']
    id_train = train_data['id']
    X_train = np.zeros((train_data.shape[0], len(p2a.columns) + 1), dtype=np.int)
    print 'training features: ' + str(X_train.shape)

    for index, row in train_data.iterrows():
        if index % 10000 == 0: print 'processed train data: ' + str(index)
        search_set = value_transformer('search_term', row['search_term'], skip_false_words)
        product_title = value_transformer('product_title', row['search_term'], skip_false_words)
        id = int(row['product_uid'])
        attrs = p2a.loc[id]
        vals = attrs.apply(lambda d: count_words(d, search_set))
        if id in p2a.index:
            X_train[index, :] = np.hstack((vals.values, count_words(product_title, search_set)))
        else:
            X_train[index, -1] = count_words(product_title, search_set)

    test_data = pd.read_csv('./dataset/test.csv')
    id_test = test_data['id']
    X_test = np.zeros((test_data.shape[0], len(p2a.columns) + 1), dtype=np.int)
    print 'test features: ' + str(X_test.shape)

    for index, row in test_data.iterrows():
        if index % 10000 == 0: print 'processed test data: ' + str(index)
        search_set = value_transformer('search_term', row['search_term'], skip_false_words)
        product_title = value_transformer('product_title', row['search_term'], skip_false_words)
        id = int(row['product_uid'])
        attrs = p2a.loc[id]
        vals = attrs.apply(lambda d: count_words(d, search_set))
        if id in p2a.index:
            X_test[index, :] = np.hstack((vals.values, count_words(product_title, search_set)))
        else:
            X_test[index, -1] = count_words(product_title, search_set)

    return X_train, y_train, X_test, id_train, id_test
