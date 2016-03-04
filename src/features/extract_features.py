import pandas as pd
import numbers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import datetime


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
    start_time = datetime.datetime.now()

    train_data = pd.read_csv('./dataset/train.csv')
    test_data = pd.read_csv('./dataset/test.csv')
    descr_by_prod = descr_by_product()
    attrs_by_prod = attrs_by_product()

    print 'Data loaded'

    numtrain = train_data.shape[0]
    data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    df = combine_all_info(data, descr_by_prod, attrs_by_prod)

    print 'Combined into one column'

    stemmer = SnowballStemmer('english')

    # remove non-alphanumeric values
    df['info'] = df['info'].str.replace(r'[^A-Za-z\d\s]+', ' ')
    print 'Non-alphanumeric removed'
    df['info'] = df['info'].str.replace(r'\s+', ' ')
    print 'Spaces trimmed'
    df['info'] = df['info'].str.replace(r'([a-z])([A-Z])', '\g<1> \g<2>')
    print 'CamelCase removed'
    df['info'] = df['info'].str.lower()
    print 'To lower done'
    df['info'] = df['info'].apply(lambda val: ' '.join([stemmer.stem(i) for i in word_tokenize(val)]))
    print 'Stemmed'

    if vect == 'cnt':
        info_vectorizer = CountVectorizer()
        st_vectorizer = CountVectorizer()
    else:
        info_vectorizer = TfidfVectorizer()
        st_vectorizer = TfidfVectorizer()

    info = info_vectorizer.fit_transform(df['info'])
    print 'Info vectorized'
    search_term = st_vectorizer.fit_transform(df['search_term'])
    print 'Search term vectorized'

    X = hstack([info, search_term])
    X_train = X.tocsc()[:numtrain,:]
    X_test = X.tocsc()[numtrain:,:]

    y_train = df.ix[:(numtrain-1), 'relevance']

    print 'Time = %s' % (datetime.datetime.now() - start_time)

    return X_train, y_train, X_test
