# import itertools as it
import os.path
import re

# import nltk.corpus as corpus
import numpy as np
import pandas as pd
import sklearn.linear_model as ln
# from nltk import PorterStemmer
from pandas import DataFrame
from sklearn import cross_validation

from src.algos.utils import RMSE_NORMALIZED
from src.features.sets import blank
from src.features.sets import boolean_columns
# from src.features.sets import get_syn
# from src.features.sets import stop_words
from src.features.extract_count_features import process_str
from src.features.extract_count_features import load_features1


def to_set(x):
    if type(x) is set: return x
    return blank


def to_str(x):
    if type(x) is str: return x
    if type(x) is unicode: return x
    return u''


def column_transformer(x, combine=True):
    if x in boolean_columns: return 'combined_boolean'
    c = str(x).lower()
    c = c.decode('utf-8', 'ignore')
    if combine:
        if re.search(
                '(:?width|height|depth|length|size|thickness|capacity|diameter|\(in\.\)|\(ft\.\)|\(mm\))',
                c) is not None:
            return 'combined_size'
        if re.search('watt', c) is not None:
            return 'combined_watt'
        if re.search('volt', c) is not None:
            return 'combined_volt'
        if re.search('(:?weight|\(lb\.\))', c) is not None:
            return 'combined_weight'
        if re.search('brand', c) is not None:
            return 'combined_brand'
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
    if c in boolean_columns:
        v = str(v).lower()
        if v.startswith('y'):
            v = c
        else:
            return ''

    av = process_str(s=v, stemming=enable_stemming, skip_stop_words=skip_stop_words)
    return av


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
        cv = value_transformer(row['name'], row['value'], skip_stop_words, enable_stemming)
        current = rs.at[id, cc]
        rs.at[id, cc] = (to_str(current) + ' ' + cv).strip()
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
        rs.at[id, 'bullet'] = (to_str(current) + ' ' +
                               value_transformer('bullet', row['product_description'],
                                                 skip_stop_words, enable_stemming)).strip()
        if is_trace_enabled: print 'bullet', id, '->', rs.at[id, 'bullet']

    print
    print 'store data into file'
    rs.to_csv(file_name, encoding='utf-8')
    print 'result:', rs.shape, '->', file_name
    return rs


def count_words(data, search):
    if type(data) is not unicode: return 0
    return len(set(data.split(' ')) & search)


def count_words_vectorized(s):
    if type(s[0]) is not unicode: return 0
    if type(s[1]) is not unicode: return 0
    return len(set(s[0].split(' ')) & set(s[1].split(' ')))


def search_in_title(s):
    if type(s[0]) is not unicode: return 0
    if type(s[1]) is not unicode: return 0
    return float(s[0] in s[1])


def search_in_bullet(s, p_to_a):
    if type(s[1]) is not unicode: return 0
    return float(s[1] in p_to_a.loc[int(s[0]), 'bullet'])


def safe_len(s):
    if type(s) is unicode: return len(s.split(' '))
    return 0


def prepare_word_set(data_file='train', product_to_trace=None, skip_stop_words=True, enable_stemming=True):
    if product_to_trace is None: product_to_trace = set([])

    file_name = './dataset/word_set.' + data_file + '.'
    if skip_stop_words:
        file_name += 'stop.'
    if enable_stemming:
        file_name += 'stemming.'
    file_name += 'csv'

    if os.path.isfile(file_name):
        print 'load', data_file, 'data from file'
        data = pd.read_csv(file_name, encoding='utf-8')
        print 'loaded', data.shape, '->', file_name
        return data

    data = pd.read_csv('./dataset/' + data_file + '.csv')
    columns = ['id', 'product_uid', 'product_title', 'search_set']
    if 'relevance' in data.columns: columns.append('relevance')
    x = DataFrame(columns=columns)

    x['id'] = data['id']
    x['product_uid'] = data['product_uid']
    if 'relevance' in data.columns:
        x['relevance'] = data['relevance']

    for index, row in data.iterrows():
        if index % 10000 == 0: print index,
        pid = int(row['product_uid'])
        oid = int(row['id'])

        is_trace_enabled = pid in product_to_trace
        if is_trace_enabled:
            print
            print 'search term', pid, '(', oid, ')', '[', row['search_term'], ']'
        x.at[index, 'search_set'] = value_transformer('search_set', row['search_term'],
                                                      skip_stop_words, enable_stemming)
        if is_trace_enabled: print 'search set', pid, '(', oid, ')', x.at[index, 'search_set']

        if is_trace_enabled: print 'product title', pid, '(', oid, ')', '[', row['product_title'], ']'
        x.at[index, 'product_title'] = value_transformer('product_title', row['product_title'],
                                                         skip_stop_words, enable_stemming)
        if is_trace_enabled: print 'product title', pid, '(', oid, ')', '[', x.at[index, 'product_title'], ']'

    print
    print 'store word set'
    x.to_csv(file_name, encoding='utf-8', index=None)
    print 'stored', x.shape, '->', file_name
    return x


def ration(s):
    if s[1] == 0: return 0
    return s[0] / s[1]


def add_ration(features):
    print 'process ration:', 'product_title',
    features['product_title/ratio/'] = features[['product_title', 'search_len']].apply(ration, axis=1,
                                                                                       reduce=True, raw=True)
    print 'search_in_title',
    features['search_in_title/ratio/'] = features[['product_title', 'search_len']].apply(ration, axis=1,
                                                                                         reduce=True, raw=True)
    print 'search_in_bullet',
    features['search_in_bullet/ratio/'] = features[['product_title', 'search_len']].apply(ration, axis=1,
                                                                                          reduce=True, raw=True)
    print 'bullet'
    features['bullet/ratio/'] = features[['product_title', 'search_len']].apply(ration, axis=1,
                                                                                reduce=True, raw=True)
    print 'ratio finish'


def add_indicators(features):
    indicators = ['combined_size', 'combined_material', 'combined_brand', 'combined_color', 'combined_weight',
                  'combined_watt', 'combined_volt', 'combined_type']
    print 'process indicators:',
    for ind in indicators:
        print ind,
        features[ind + '/indicator/'] = features[ind].apply(lambda x: float(x > 0))


def integrate_with_anton(features):
    print 'integrate with anton'
    anton_train = pd.read_csv('./dataset/good_ft_2_train.csv', index_col='id')
    anton_test = pd.read_csv('./dataset/good_ft_2_test.csv', index_col='id')
    if 'relevance' in features.columns:
        anton = anton_train
        features = features[features.columns[features.columns != 'relevance']]
    else:
        anton = anton_test
    features = pd.merge(anton, features, right_on='id', left_index=True)
    print 'finish integrating'
    return features


def match_features(p_to_a=None, data_file='train'):
    file_name = './dataset/raw_features.' + data_file + '.csv'

    if os.path.isfile(file_name):
        print 'load', data_file, 'data from file'
        features = pd.read_csv(file_name)
        print 'loaded', features.shape, '->', file_name
        return features

    if p_to_a is None: p_to_a = product2attrs()
    data = prepare_word_set(data_file)
    print 'build start data frame'
    attrs = p_to_a.columns

    features = DataFrame(columns=attrs)
    features['id'] = data['id']
    if 'relevance' in data.columns: features['relevance'] = data['relevance']

    print 'calculate search_len'
    features['search_len'] = data['search_set'].apply(safe_len)
    print 'calculate product_title'
    features['product_title'] = data[['product_title', 'search_set']].apply(count_words_vectorized, axis=1,
                                                                            reduce=True, raw=True)
    print 'calculate search_in_title'
    features['search_in_title'] = data[['search_set', 'product_title']].apply(search_in_title, axis=1,
                                                                              reduce=True, raw=True)
    print 'calculate search_in_bullet'
    features['search_in_bullet'] = data[['product_uid', 'search_set']].apply(lambda s: search_in_bullet(s, p_to_a)
                                                                             , axis=1, reduce=True, raw=True)

    print 'process attributes'
    features = features.fillna(0.0)

    tmp = np.zeros((data.shape[0], len(attrs)))
    for index, row in data.iterrows():
        if index % 10000 == 0: print index,
        pid = row['product_uid']
        search_set = row['search_set']
        if type(search_set) is not unicode: continue
        search_set = set(search_set.split(' '))
        values = p_to_a.loc[pid]
        tmp[index] = values.apply(lambda d: count_words(d, search_set))

    print
    print 'integrate features with attributes'
    features[attrs] = tmp
    add_ration(features)
    add_indicators(features)
    features = integrate_with_anton(features)
    print 'store features'
    features.to_csv(file_name, index=None)
    print 'stored', features.shape, '->', file_name
    return features


def features_to_x(features):
    columns = features.columns
    columns = columns[np.all([columns[:] != 'relevance',
                              columns[:] != 'id'], axis=0)]
    x = features[columns]
    return x


def zero_normalization(features, merge=True):
    file_name = './dataset/zero_normalization.csv'

    if os.path.isfile(file_name):
        print 'load', file_name, 'data from file'
        indexes = pd.Series.from_csv(file_name)
        if 'relevance' not in features.columns:
            indexes = indexes[indexes.index != 'relevance']
        print 'loaded', indexes.shape, '->', file_name
    else:
        if 'relevance' not in features.columns: raise Exception('process train features before test')
        indexes = features.apply(np.sum, axis=0) > 0
        print 'store indexes'
        indexes.to_csv(file_name)
        print 'stored', indexes.shape, '->', file_name

    if merge:
        features = features.copy(deep=True)
        features['bullet'] += features[features.columns[indexes == False]].apply(np.sum, axis=1)

    features = features[features.columns[indexes]]
    print 'zero normalized', features.shape
    return features


def fill_start_mask(mask, start_mask):
    if start_mask is None: return mask
    if type(start_mask) is str:
        file_name = './dataset/mask.' + start_mask + '.csv'
        if not os.path.isfile(file_name): raise Exception('can not find start mask')
        print 'load', file_name, 'data from file'
        start_mask = pd.Series.from_csv(file_name)
        print 'loaded', start_mask.shape, '->', file_name
        for col in mask.index:
            if col in start_mask.index:
                mask[col] = start_mask[col]
        print 'start mask applied'
        return mask

    for ix in mask.index:
        if ix in start_mask: mask[ix] = 'F'
    return mask


def select_features(mask_name, features, cls=ln.LinearRegression(normalize=True), allow_merge=False, start_mask=None):
    features = features.copy(deep=True)
    file_name = './dataset/mask.' + mask_name + '.csv'
    changes = 0
    if os.path.isfile(file_name):
        print 'load', file_name, 'data from file'
        mask = pd.Series.from_csv(file_name)
        if 'relevance' not in features.columns:
            mask = mask[mask.index != 'relevance']
        print 'loaded', mask.shape, '->', file_name

        col_to_merge = features.columns[mask == 'M']
        if len(col_to_merge) > 0:
            features['bullet'] += features[col_to_merge].apply(np.sum, axis=1)

        result_features = features[features.columns[mask == 'F']]
        return result_features
    print 'source', features.shape
    mask = features.loc[features.index[0]].apply(lambda x: 'D')
    if 'relevance' in mask.index: mask['relevance'] = 'F'
    if 'id' in mask.index: mask['id'] = 'F'

    mask = fill_start_mask(mask, start_mask)

    y = features['relevance']
    if len(features_to_x(features[features.columns[mask == 'F']]).columns) > 0:
        score = cross_validation.cross_val_score(cls, features_to_x(features[features.columns[mask == 'F']])
                                                 , y, scoring=RMSE_NORMALIZED, cv=5).mean()
    else:
        score = -100000
    print 'start score', score
    for i, feature in enumerate(mask.index[mask == 'D']):
        print 'add', feature,
        mask[feature] = 'F'
        filtered = features[features.columns[mask == 'F']]
        print filtered.shape
        s = cross_validation.cross_val_score(cls, features_to_x(filtered), y, scoring=RMSE_NORMALIZED, cv=5).mean()
        print 'calculated score', s,
        if s > score:
            score = s
            changes += 1
            print 'accept feature', feature
        else:
            mask[feature] = 'D'
            print 'reject feature', feature

    print 'remove features', score
    for feature in mask.index[mask == 'F']:
        if feature in {'relevance', 'id'}: continue
        print 'remove', feature,
        mask[feature] = 'D'
        filtered = features[features.columns[mask == 'F']]
        print filtered.shape
        s = cross_validation.cross_val_score(cls, features_to_x(filtered), y, scoring=RMSE_NORMALIZED, cv=5).mean()
        print 'calculated score', s,
        if s > score:
            score = s
            changes += 1
            print 'reject feature', feature
        else:
            mask[feature] = 'F'
            print 'rollback feature', feature

    if allow_merge:
        print 'merge features', score
        unmerged = {'relevance', 'id', 'bullet', 'search_len', 'synonyms'}
        for i, feature in enumerate(mask.index):
            if feature in unmerged: continue
            print 'merge', feature,
            backup = mask[feature]
            mask[feature] = 'M'
            features['bullet'] += features[feature]
            filtered = features[features.columns[mask == 'F']]
            print filtered.shape
            s = cross_validation.cross_val_score(cls, features_to_x(filtered), y, scoring=RMSE_NORMALIZED, cv=5).mean()
            print 'calculated score', s,
            if s > score:
                score = s
                print 'merge feature', feature
            elif (s == score) and (backup == 'D'):
                print 'merge feature', feature
            else:
                mask[feature] = backup
                features['bullet'] -= features[feature]
                print 'rollback feature', feature

    result_features = features[features.columns[mask == 'F']]
    print 'store', mask.shape, '->', file_name
    mask.to_csv(file_name)
    print 'result score', score, result_features.shape, 'changes', changes
    return result_features


def apply_search_len(s):
    if s['search_len'] == 0: return 0
    return float(s[0]) / s['search_len']


def normalize_search_len(features):
    features = features.copy()
    print 'normalization by search_len', features.shape, '->',
    for i, col in enumerate(features.columns):
        if col in {'id', 'relevance', 'search_len'}: continue
        features[col + '/slennorm/'] = features[[col, 'search_len']].apply(apply_search_len, axis=1)
    print features.shape
    return features
