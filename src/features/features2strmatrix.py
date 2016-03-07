import re

import pandas as pd
from pandas import DataFrame


def column_transformer(x):
    c = str(x).lower()
    c = re.sub('\s', '_', c)
    c = re.sub('bullet\d+', 'bullet', c)
    return c


def value_transformer(c, v):
    v = str(v).lower()
    av = set(re.split('[\s,]', v))
    return av


def product2attrs():
    attrs = pd.read_csv('./dataset/attributes.csv')
    attrs = attrs[attrs['product_uid'] == attrs['product_uid']]
    print 'attributes: ' + str(attrs.shape)
    colls = attrs['name'].apply(column_transformer).unique()

    product_ids = [int(x) for x in attrs['product_uid'].unique()]
    rs = DataFrame(index=product_ids, columns=colls)

    for index, row in attrs.iterrows():
        if index % 100000 == 0: print 'processed: ' + str(index)
        cc = column_transformer(row['name'])
        cv = value_transformer(cc, row['value'])
        id = int(row['product_uid'])
        current = rs.at[id, cc]
        if type(current) is float:
            rs.at[id, cc] = cv
        else:
            rs.at[id, cc] = current | cv

    print 'result:' + str(rs.shape)
    return rs
