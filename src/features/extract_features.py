import pandas as pd
import numbers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


def make_string(s):
    if isinstance(s, numbers.Number):
        return str(s)
    return s


def attrs_by_product():
    attrs = pd.read_csv('./dataset/attributes.csv')
    attrs['attrs'] = attrs['name'].astype(str) + ' ' + attrs['value'].astype(str)
    attrs_by_product_id = attrs.groupby('product_uid')['attrs'].apply(lambda x: ' '.join(make_string(x)))
    return attrs_by_product_id


def descr_by_product():
    descr = pd.read_csv('./dataset/product_descriptions.csv', index_col='product_uid')
    return descr['product_description']


def combine_all_info(data, descr_by_prod, attrs_by_prod):
    data['descr'] = data.apply(lambda row: descr_by_prod[row['product_uid']], axis=1)
    data['attrs'] = data.apply(lambda row: attrs_by_prod[row['product_uid']] if row['product_uid'] in attrs_by_prod else '', axis=1)
    data['info'] = data['product_title'].astype(str) + ' ' + data['descr'].astype(str) + ' ' + data['attrs'].astype(str)
    data = data.drop(['product_title', 'descr', 'attrs'], axis=1)
    return data


def get_all_in_one_feature_matrix(vect='cnt'):
    train_data = pd.read_csv('./dataset/train.csv')
    test_data = pd.read_csv('./dataset/test.csv')
    descr_by_prod = descr_by_product()
    attrs_by_prod = attrs_by_product()

    train_df = combine_all_info(train_data, descr_by_prod, attrs_by_prod)
    test_df = combine_all_info(test_data, descr_by_prod, attrs_by_prod)

    # remove non-alphanumeric values
    train_df['info'] = train_df['info'].str.replace(r'[^A-Za-z\d\s]+', ' ')
    train_df['info'] = train_df['info'].str.replace(r'\s+', ' ')
    test_df['info'] = test_df['info'].str.replace(r'[^A-Za-z\d\s]+', ' ')
    test_df['info'] = test_df['info'].str.replace(r'\s+', ' ')

    if vect == 'cnt':
        info_vectorizer = CountVectorizer()
        st_vectorizer = CountVectorizer()
    else:
        info_vectorizer = TfidfVectorizer()
        st_vectorizer = TfidfVectorizer()

    info_vectorizer.fit(pd.concat([train_df['info'], test_df['info']], axis=0))
    st_vectorizer.fit(pd.concat([train_df['search_term'], test_df['search_term']], axis=0))

    X_train = hstack([info_vectorizer.transform(train_df['info']), st_vectorizer.transform(train_df['search_term'])])
    y_train = train_df['relevance']
    X_test = hstack([info_vectorizer.transform(test_df['info']), st_vectorizer.transform(test_df['search_term'])])

    return X_train, y_train, X_test
