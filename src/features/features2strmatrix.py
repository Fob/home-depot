import re

import numpy as np
import pandas as pd
from nltk import PorterStemmer
from nltk.corpus import stopwords
from pandas import DataFrame

stop_words = set(stopwords.words('english')) | {'', '&', 'x', '&amp;', 'in.'}


def column_transformer(x, combine_sizes=True):
    c = str(x).lower()
    c = re.sub('\s', '_', c)
    if combine_sizes:
        if re.search('(:?width|height|depth|length)', c) is not None:
            return 'combined_size'
    c = re.sub('bullet\d+', 'bullet', c)
    return c


def value_transformer(c, v, skip_stop_words=True):
    v = str(v).lower()
    v = v.decode('utf-8', 'ignore')
    v = re.sub('(?<!\d)\.(?!\d)', ' ', v)
    av = set(re.split('[\s,\)\(]', v))
    if skip_stop_words:
        av = av - stop_words
    stemmer = PorterStemmer()
    av = set([stemmer.stem(w) for w in av])
    return av


def product2attrs(product_to_trace=None, combine_sizes=True, skip_stop_words=True):
    if product_to_trace is None: product_to_trace = {}
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
        id = int(row['product_uid'])
        cc = column_transformer(row['name'], combine_sizes)
        is_trace_enabled = id in product_to_trace

        if is_trace_enabled: print row['name'], ' ', id, '->', row['value']
        cv = value_transformer(cc, row['value'], skip_stop_words)
        current = rs.at[id, cc]
        if type(current) is float:
            rs.at[id, cc] = cv
        else:
            rs.at[id, cc] = current | cv
        if is_trace_enabled: print cc, ' ', id, '->', rs.at[id, cc]

    print 'descriptions :' + str(descrs.shape)

    for index, row in descrs.iterrows():
        if index % 10000 == 0: print 'processed descr: ' + str(index)
        id = int(row['product_uid'])
        if id not in rs.index: continue
        is_trace_enabled = id in product_to_trace

        if is_trace_enabled: print 'product_description ', id, '->', row['product_description']
        rs.at[id, 'full_descr'] = value_transformer('full_descr', row['product_description'], skip_stop_words)
        if is_trace_enabled: print 'full_descr ', id, '->', rs.at[id, 'full_descr']

    print 'result:' + str(rs.shape)
    return rs


def count_words(data, search):
    if type(data) is float:
        return 0
    return len(data & search)


def internal_enrich_features(data, product_to_trace, id_to_trace, skip_stop_words, p2a):
    x = np.zeros((data.shape[0], len(p2a.columns) + 1), dtype=np.int)
    column_names = np.hstack((p2a.columns, 'product_title'))
    for index, row in data.iterrows():
        if index % 10000 == 0: print 'processed data: ', index
        pid = int(row['product_uid'])
        oid = int(row['id'])
        is_trace_enabled = (pid in product_to_trace) or (oid in id_to_trace)

        if is_trace_enabled: print 'search term', pid, '(', oid, ')', '[', row['search_term'], ']'
        search_set = value_transformer('search_term', row['search_term'], skip_stop_words)
        if is_trace_enabled: print 'search set', pid, '(', oid, ')', '[', search_set, ']'

        if is_trace_enabled: print 'product title', pid, '(', oid, ')', '[', row['product_title'], ']'
        product_title = value_transformer('product_title', row['product_title'], skip_stop_words)
        if is_trace_enabled: print 'product title', pid, '(', oid, ')', '[', product_title, ']'

        attrs = p2a.loc[pid]
        vals = attrs.apply(lambda d: count_words(d, search_set))
        if pid in p2a.index:
            x[index, :] = np.hstack((vals.values, count_words(product_title, search_set)))
        else:
            x[index, -1] = count_words(product_title, search_set)
        if is_trace_enabled:
            print 'result', pid, '(', oid, ')'
            print list(column_names[x[index, :] > 0])
            print list(x[index, x[index, :] > 0])

    print 'feature prepared ', x.shape
    return x


def load_features(product_to_trace=None, id_to_trace=None, combine_sizes=True, skip_stop_words=True):
    if id_to_trace is None: id_to_trace = {}
    if product_to_trace is None: product_to_trace = {}
    p2a = product2attrs(product_to_trace, combine_sizes, skip_stop_words)
    train_data = pd.read_csv('./dataset/train.csv')
    y_train = train_data['relevance']
    id_train = train_data['id']
    print 'preparing training features: ', train_data.shape
    X_train = internal_enrich_features(train_data, product_to_trace, id_to_trace, skip_stop_words, p2a)

    test_data = pd.read_csv('./dataset/test.csv')
    id_test = test_data['id']
    print 'preparing test features: ' + str(test_data.shape)
    X_test = internal_enrich_features(test_data, product_to_trace, id_to_trace, skip_stop_words, p2a)

    return X_train, y_train, X_test, id_train, id_test

# X_train, y_train, X_test, id_train, id_test = load_features(product_to_trace={100001})
