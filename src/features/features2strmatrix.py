import itertools as it
import re

import nltk.corpus as corpus
import numpy as np
import pandas as pd
from nltk import PorterStemmer
from nltk.corpus import stopwords
from pandas import DataFrame

stop_words = set(stopwords.words('english')) | {'', '&', 'x', 'ft', 'lb', 'f'}


def column_transformer(x, combine=True):
    c = str(x).lower()
    if combine:
        if re.search('(:?width|height|depth|length|size|thickness|capacity)', c) is not None:
            return 'combined_size'
        if re.search('weight', c) is not None:
            return 'combined_weight'
        if re.search('color', c) is not None:
            return 'combined_color'
        if re.search('material', c) is not None:
            return 'combined_material'
        if re.search('temperature', c) is not None:
            return 'combined_temperature'
    c = re.sub('bullet\d+', 'bullet', c)
    return c


def value_transformer(c, v, skip_stop_words=True):
    v = str(v).lower()
    v = v.decode('utf-8', 'ignore')
    v = re.sub('(?<!\d)\.(?!\d)', ' ', v)
    v = re.sub('(?<!\d)/(?!\d)', ' ', v)
    v = re.sub('&\w+;', ' ', v)
    av = set(re.split('[\s,\)\(\xb0]', v))
    if skip_stop_words:
        av = av - stop_words
    stemmer = PorterStemmer()
    av = set([stemmer.stem(w) for w in av])
    return av


def search_transformer(v, skip_stop_words=True):
    v = str(v).lower()
    v = v.decode('utf-8', 'ignore')
    v = re.sub('(?<!\d)\.(?!\d)', ' ', v)
    av = set(re.split('[\s,\)\(\xb0]', v))
    if skip_stop_words: av = av - stop_words

    wn = corpus.wordnet
    avs = set([])
    for w in av:
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


def product2attrs(product_to_trace=None, combine=True, skip_stop_words=True):
    if product_to_trace is None: product_to_trace = {}
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

    for index, row in attrs.iterrows():
        if index % 100000 == 0: print 'processed:', index
        id = int(row['product_uid'])
        cc = column_transformer(row['name'], combine)
        is_trace_enabled = id in product_to_trace

        if is_trace_enabled: print row['name'], id, '->', row['value']
        cv = value_transformer(cc, row['value'], skip_stop_words)
        current = rs.at[id, cc]
        if type(current) is float:
            rs.at[id, cc] = cv
        else:
            rs.at[id, cc] = current | cv
        if is_trace_enabled: print cc, id, '->', rs.at[id, cc]

    print 'descriptions :', descrs.shape

    for index, row in descrs.iterrows():
        if index % 10000 == 0: print 'processed descr:', index
        id = int(row['product_uid'])
        if id not in rs.index: continue
        is_trace_enabled = id in product_to_trace

        if is_trace_enabled: print 'product_description', id, '->', row['product_description']
        current = rs.at[id, 'bullet']
        if type(current) is float:
            rs.at[id, 'bullet'] = value_transformer('bullet', row['product_description'], skip_stop_words)
        else:
            rs.at[id, 'bullet'] = current | value_transformer('bullet', row['product_description'], skip_stop_words)
        if is_trace_enabled: print 'bullet', id, '->', rs.at[id, 'bullet']

    print 'result:', rs.shape
    return rs


def count_words(data, search):
    if type(data) is float:
        return 0
    return len(data & search)


def internal_enrich_features(data, product_to_trace, id_to_trace, skip_stop_words, p2a):
    attrs_len = len(p2a.columns)
    x = np.zeros((data.shape[0], attrs_len + 1), dtype=np.int)
    column_names = np.hstack((p2a.columns, 'syn_combo'))
    for index, row in data.iterrows():
        if index % 10000 == 0: print 'processed data:', index
        pid = int(row['product_uid'])
        oid = int(row['id'])
        is_trace_enabled = (pid in product_to_trace) or (oid in id_to_trace)

        if is_trace_enabled: print 'search term', pid, '(', oid, ')', '[', row['search_term'], ']'
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
            x[index, -1] += np.sum(attrs.apply(lambda d: count_words(d, syn_set)))

        # title to bullet
        x[index, 0] += count_words(product_title, search_set)
        # title to synonyms combo
        x[index, -1] += count_words(product_title, syn_set)

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
    print 'preparing training features:', train_data.shape
    X_train = internal_enrich_features(train_data, product_to_trace, id_to_trace, skip_stop_words, p2a)

    test_data = pd.read_csv('./dataset/test.csv')
    id_test = test_data['id']
    print 'preparing test features:', test_data.shape
    X_test = internal_enrich_features(test_data, product_to_trace, id_to_trace, skip_stop_words, p2a)

    return X_train, y_train, X_test, id_train, id_test

# X_train, y_train, X_test, id_train, id_test = load_features(product_to_trace={100001})
