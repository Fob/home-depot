import src.features.extract_features as ext_ft
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def stemming_row(row_string):
    import re
    from nltk import PorterStemmer
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    words = [w for w in re.sub("[^A-Za-z-\s\(\)]", "", row_string).split() if w not in stop]
    return ' '.join([PorterStemmer().stem_word(s) for s in words])

def create_and_save_all_in_one_feature_matrix_with_stemming():
    train_data = pd.read_csv('./dataset/train.csv')
    test_data = pd.read_csv('./dataset/test.csv')
    descr_by_prod = ext_ft.descr_by_product()
    attrs_by_prod = ext_ft.attrs_by_product()

    train_df = ext_ft.combine_all_info(train_data, descr_by_prod, attrs_by_prod)
    test_df = ext_ft.combine_all_info(test_data, descr_by_prod, attrs_by_prod)

    train_df['info'] = [stemming_row(row_info) for row_info in train_df['info']]
    train_df['search_term'] = [stemming_row(row_info) for row_info in train_df['search_term']]
    test_df['info'] = [stemming_row(row_info) for row_info in test_df['info']]
    test_df['search_term'] = [stemming_row(row_info) for row_info in test_df['search_term']]

    train_df.to_csv('./dataset/train_all_info_with_stemming.csv', index=None)
    test_df.to_csv('./dataset/test_all_info_with_stemming.csv', index=None)


def get_all_in_one_feature_matrix_with_stemming(vect='cnt'):
    train_df = pd.read_csv('./dataset/train_all_info_with_stemming.csv')
    test_df = pd.read_csv('./dataset/test_all_info_with_stemming.csv')

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
