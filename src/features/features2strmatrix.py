import itertools as it
import os.path
import re

import nltk.corpus as corpus
import numpy as np
import pandas as pd
import scipy.sparse as sci
from nltk import PorterStemmer
from pandas import DataFrame
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer

from src.algos.utils import rmse
from src.features.sets import boolean_columns
from src.features.sets import get_syn
from src.features.sets import stop_words


def to_set(x):
    if type(x) is float: return set([])
    return x


def column_transformer(x, combine=True):
    if x in boolean_columns: return 'combined_boolean'
    c = str(x).lower()
    c = c.decode('utf-8', 'ignore')
    if combine:
        if re.search(
                '(:?width|height|depth|length|size|thickness|capacity|diameter|\(in\.\)|\(ft\.\)|\(mm\)|\(miles\))',
                c) is not None:
            return 'combined_size'
        if re.search('(:?weight|\(lb\.\))', c) is not None:
            return 'combined_weight'
        if re.search('(:?color|rgb value)', c) is not None:
            return 'combined_color'
        if re.search('material', c) is not None:
            return 'combined_material'
        if re.search('temperature', c) is not None:
            return 'combined_temperature'
        if re.search('type', c) is not None:
            return 'combined_type'
        if re.search('(:?time|\(hours\)|\(min\.\))', c) is not None:
            return 'combined_time'
        if re.search('(:?number of|count|# of)', c) is not None:
            return 'combined_count'
        if re.search('(?:degree|angle|\(deg\))', c) is not None:
            return 'combined_angle'
        if re.search('(:?\(sq\.? ft\.?\)|square foot|\(sq\. in\.\))', c) is not None:
            return 'combined_area'
        if re.search('(?:\(\w?hz\)|frequency|\(rpm\)|\(gpm\)|\(cfm\)|\(mph\)|speed|/hour|/min|per min)', c) is not None:
            return 'combined_speed'
        if re.search('(?:\(db\)|noise)', c) is not None:
            return 'combined_noise'
        if re.search('\([^\(\)]+\)$', c) is not None:
            return 'combined_measurements'
        if re.search('/', c) is not None:
            return 'combined_type'

    c = re.sub('bullet\d+', 'bullet', c)
    return c


def value_transformer(c, v, skip_stop_words=True, enable_stemming=True):
    v = str(v).lower()
    if c in boolean_columns:
        if v.startswith('y'):
            v = str(c).lower()
        else:
            return set([])
    v = v.decode('utf-8', 'ignore')
    v = re.sub('(?<!\d)\.(?!\d)', ' ', v)
    v = re.sub('(?<!\d)/(?!\d)', ' ', v)
    v = re.sub('&\w+;', ' ', v)
    av = set(re.split('[\s,\)\(\xb0\?]', v))
    if skip_stop_words:
        av = av - stop_words
    if enable_stemming:
        stemmer = PorterStemmer()
        av = set([stemmer.stem(w) for w in av])
    return av


def search_transformer(v, skip_stop_words=True):
    v = str(v).lower()
    v = v.decode('utf-8', 'ignore')
    v = re.sub('(?<!\d)\.(?!\d)', ' ', v)
    av = set(re.split('[\s,\)\(\xb0]', v))
    # if skip_stop_words: av = av - stop_words

    wn = corpus.wordnet
    avs = set([])
    for w in av:
        avs = avs | get_syn(w)
        synonyms = wn.synsets(w)
        if len(synonyms) > 0:
            ws = set(it.chain.from_iterable(ww.lemma_names() for ww in synonyms))
            avs = avs | ws
    if skip_stop_words: avs = avs - stop_words - av

    stemmer = PorterStemmer()
    av = set([stemmer.stem(w) for w in av])
    if skip_stop_words: av = av - stop_words

    avs = set([stemmer.stem(w) for w in avs])
    if skip_stop_words: avs = avs - stop_words - av
    return av, avs


def serialize_attr(x):
    if type(x) is float: return None
    str_val = ''
    for y in x:
        str_val += y + ' '
    return str_val.strip(' ')


def deserialize_attr(x):
    if type(x) is unicode:
        return set(x.split(' '))
    return x


def product2attrs(product_to_trace=None, combine=True, skip_stop_words=True, enable_stemming=True):
    if product_to_trace is None: product_to_trace = {}
    file_name = './dataset/product_to_attributes.'
    if combine: file_name += 'combined.'
    if skip_stop_words: file_name += 'stop.'
    if enable_stemming: file_name += 'stemming.'
    file_name += 'csv'

    if os.path.isfile(file_name):
        print 'load data from file'
        rs = pd.read_csv(file_name, encoding='utf-8', index_col=0)
        rs = rs.applymap(deserialize_attr)
        print 'loaded', file_name, '->', rs.shape
        return rs

    attrs = pd.read_csv('./dataset/attributes.csv')
    attrs = attrs[attrs['product_uid'] == attrs['product_uid']]
    descrs = pd.read_csv('./dataset/product_descriptions.csv')
    descrs = descrs[descrs['product_uid'] == descrs['product_uid']]

    print 'attributes:', attrs.shape
    colls = attrs['name'].apply(lambda c: column_transformer(c, combine)).unique()
    print 'attributes columns:', len(colls)

    # noinspection PyUnresolvedReferences
    product_ids = [int(x) for x in pd.concat([attrs['product_uid'], descrs['product_uid']]).unique()]
    print 'unique ids:', len(product_ids)

    rs = DataFrame(index=product_ids, columns=np.hstack(colls))
    rs.index.names = ['product_uid']

    print 'process attrs'
    for index, row in attrs.iterrows():
        if index % 100000 == 0: print index,
        id = int(row['product_uid'])
        cc = column_transformer(row['name'], combine)
        is_trace_enabled = id in product_to_trace

        if is_trace_enabled:
            print
            print row['name'], id, '->', row['value']
        cv = value_transformer(row['name'], row['value'], skip_stop_words)
        current = rs.at[id, cc]
        rs.at[id, cc] = to_set(current) | cv
        if is_trace_enabled: print cc, id, '->', rs.at[id, cc]

    print
    print 'descriptions :', descrs.shape

    for index, row in descrs.iterrows():
        if index % 10000 == 0: print index,
        id = int(row['product_uid'])
        if id not in rs.index: continue
        is_trace_enabled = id in product_to_trace

        if is_trace_enabled:
            print
            print 'product_description', id, '->', row['product_description']
        current = rs.at[id, 'bullet']
        rs.at[id, 'bullet'] = to_set(current) | value_transformer('bullet',
                                                                  row['product_description'], skip_stop_words)
        if is_trace_enabled: print 'bullet', id, '->', rs.at[id, 'bullet']

    print
    print 'store data into file'
    serialized_data = rs.applymap(serialize_attr)
    serialized_data.to_csv(file_name, encoding='utf-8')
    print 'result:', rs.shape, '->', file_name
    return rs


def count_words(data, search):
    if type(data) is set:
        return len(data & search)
    return 0


def internal_enrich_features(data, product_to_trace, skip_stop_words, p2a):
    attrs_len = len(p2a.columns)
    x = np.zeros((data.shape[0], attrs_len), dtype=np.float)
    x_derivatives = np.zeros((data.shape[0], 4), dtype=np.float)
    column_names = p2a.columns
    for index, row in data.iterrows():
        if index % 10000 == 0: print index,
        pid = int(row['product_uid'])
        oid = int(row['id'])
        is_trace_enabled = pid in product_to_trace

        if is_trace_enabled:
            print
            print 'search term', pid, '(', oid, ')', '[', row['search_term'], ']'
        search_set, syn_set = search_transformer(row['search_term'], skip_stop_words)
        if is_trace_enabled: print 'search set', pid, '(', oid, ')', search_set
        if is_trace_enabled: print 'syn set', pid, '(', oid, ')', syn_set

        if is_trace_enabled: print 'product title', pid, '(', oid, ')', '[', row['product_title'], ']'
        product_title = value_transformer('product_title', row['product_title'], skip_stop_words)
        if is_trace_enabled: print 'product title', pid, '(', oid, ')', '[', product_title, ']'

        if pid in p2a.index:
            attrs = p2a.loc[pid]
            vals = attrs.apply(lambda d: count_words(d, search_set))
            x[index, :attrs_len] = vals.values
            # synonyms combo
            x_derivatives[index, 0] += np.sum(attrs.apply(lambda d: count_words(d, syn_set)))

        # title to bullet
        x[index, 4] += count_words(product_title, search_set)
        # title to synonyms combo
        x_derivatives[index, 0] += count_words(product_title, syn_set)
        x_derivatives[index, 1] = len(search_set)
        if x_derivatives[index, 1] != 0:
            x_derivatives[index, 2] = float(np.sum(x[index, :])) / float(len(search_set))

        if is_trace_enabled:
            print 'result', pid, '(', oid, ')'
            print list(column_names[x[index, :] > 0])
            print list(x[index, x[index, :] > 0])
            print list(x_derivatives[index, :])

    print
    print 'feature prepared', x.shape, x_derivatives.shape
    return x, x_derivatives


def merge_word_match(x_train, x_test, merge_factor):
    x_train_merged = np.array(x_train)
    x_test_merged = np.array(x_test)

    indx = np.sum(x_train_merged, axis=0) < merge_factor
    print 'merge columns', x_train_merged[:, indx].shape
    x_train_merged[:, 0] = x_train_merged[:, 0] + np.sum(x_train_merged[:, indx], axis=1)
    x_train_merged = x_train_merged[:, indx == False]

    x_test_merged[:, 0] = x_test_merged[:, 0] + np.sum(x_test_merged[:, indx], axis=1)
    x_test_merged = x_test_merged[:, indx == False]
    return x_train_merged, x_test_merged


def load_raw_features(product_to_trace, skip_stop_words, p2a):
    train_data = pd.read_csv('./dataset/train.csv')
    y_train = train_data['relevance']
    id_train = train_data['id']
    print 'preparing training features:', train_data.shape
    x_train, x_tr_der = internal_enrich_features(train_data, product_to_trace, skip_stop_words, p2a)

    test_data = pd.read_csv('./dataset/test.csv')
    id_test = test_data['id']
    print 'preparing test features:', test_data.shape
    x_test, x_ts_der = internal_enrich_features(test_data, product_to_trace, skip_stop_words, p2a)

    return x_train, x_tr_der, y_train, x_test, x_ts_der, id_train, id_test


def load_features(merge_factor=20, product_to_trace=None, skip_stop_words=True, p2a=None):
    if product_to_trace is None: product_to_trace = {}
    if p2a is None: p2a = product2attrs()

    x_train, x_tr_der, y_train, x_test, x_ts_der, id_train, id_test = load_raw_features(product_to_trace,
                                                                                        skip_stop_words,
                                                                                        p2a)

    x_train, x_test = merge_word_match(x_train, x_test, merge_factor)
    print 'after merge', x_train.shape, x_test.shape
    x_train = np.hstack((x_train, x_tr_der))
    x_test = np.hstack((x_test, x_ts_der))

    return x_train, y_train, x_test, id_train, id_test


def product_search_vectorize(product_vectorizer, search_term_vectorizer, data):
    x_product = product_vectorizer.transform([str(p) for p in data['product_uid']])
    x_search = search_term_vectorizer.transform(data['search_term'])
    x = sci.hstack((x_product, x_search))
    print 'features prepared', x.shape
    return x


def product_to_search_features(p2a):
    train_data = pd.read_csv('./dataset/train.csv')
    test_data = pd.read_csv('./dataset/test.csv')

    products = [str(p) for p in p2a.index]
    product_vectorizer = CountVectorizer()
    product_vectorizer.fit(products)
    search_term_vectorizer = CountVectorizer()
    search_term_vectorizer.fit(test_data['search_term'])

    y_train = train_data['relevance']
    id_train = train_data['id']
    x_train = product_search_vectorize(product_vectorizer, search_term_vectorizer, train_data)

    id_test = test_data['id']
    x_test = product_search_vectorize(product_vectorizer, search_term_vectorizer, test_data)
    return x_train, y_train, x_test, id_train, id_test


def product_search_regression(p2a, a=100000.0):
    print 'product to search regression'
    x_train, y_train, x_test, id_train, id_test = product_to_search_features(p2a=p2a)

    clf = linear_model.ElasticNet(alpha=a, normalize=True)
    clf.fit(x_train, y_train)

    y_predicted = clf.predict(x_train)

    y_test = clf.predict(x_test)

    print 'regression train reult', rmse(y_train, y_predicted), y_predicted.shape, y_test.shape

    return y_predicted, y_test

# x_train, x_tr_der, y_train, x_test, x_ts_der, id_train, id_test = load_features(product_to_trace={100001})
