import pandas as pd
import numpy as np
import numbers
import re
from nltk.stem.porter import *
from nltk.corpus import stopwords
import datetime


stemmer = PorterStemmer()
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}
stop = stopwords.words('english')


def process_str(s):
    if isinstance(s, numbers.Number):
        return str(s)
    elif isinstance(s, str):
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)

        #todo: somehow process dimensions
        # 48 in. W x 72 in. H x 18 in. D  // dimension format
        s = re.sub(r" W\.?\b", "", s)
        s = re.sub(r" L\.?\b", "", s)
        s = re.sub(r" H\.?\b", "", s)
        s = re.sub(r" Dia\.?\b", "", s)

        s = s.lower()

        s = re.sub(r"(\d)-(\d)", r"\1_\2", s)
        s = s.replace("-", " ")
        s = s.replace("(", " ")
        s = s.replace(")", " ")
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace(":"," ")
        s = s.replace("..",".")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")

        s = s.replace("//","/")
        s = s.replace(" / "," ")
        s = re.sub(r"(\D)/", r"\1 ", s)
        s = re.sub(r"/(\D)", r" \1", s)

        s = s.replace("&amp;", " and ")
        s = s.replace("&#39;s", "")
        s = s.replace("#", " ")

        s = re.sub(r"([0-9]),([0-9])", r"\1\2", s)
        s = s.replace(",", " ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)

        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)

        s = s.replace(" x "," ") #xbi
        s = s.replace("*"," ") #xbi
        s = s.replace(" by "," ") #xbi

        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')( *)\.?", r"\1 in. ", s)
        s = re.sub(r"([0-9]+)( *)(yards|yard|yd|yds)( *)\.?", r"\1 yds. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')( *)\.?", r"\1 ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)( *)\.?", r"\1 lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq)( *)\.?( *)(feet|foot|ft)( *)\.?", r"\1 sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu)( *)\.?( *)(feet|foot|ft)( *)\.?", r"\1 cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)( *)\.?", r"\1 gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)( *)\.?", r"\1 oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)( *)\.?", r"\1 cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)( *)\.?", r"\1 mm. ", s)
        #s = s.replace(""," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)( *)\.?", r"\1 deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)( *)\.?", r"\1 volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)( *)\.?", r"\1 watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)( *)\.?", r"\1 amp. ", s)

        s = s.replace(" . "," ")
        s = re.sub(r"\s+", " ", s)

        s = " ".join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = s.decode('utf-8', 'ignore') #todo
        s = " ".join([stemmer.stem(w) for w in s.split(" ")])

        return s
    else:
        return ""


def make_string(s):
    if isinstance(s, numbers.Number):
        return str(s)
    return s


def descr_by_product():
    descr = pd.read_csv('./dataset/product_descriptions.csv', index_col='product_uid')
    return descr['product_description']


def extract_attribute_map(source_df, attr_name):
    is_attr = np.array([1 if attr_name in str(x).lower() else 0 for x in source_df['name']])
    attrs_df = source_df[is_attr > 0].copy()
    attrs_df['value'] = attrs_df['value'].str.lower()
    attrs_df['value'] = attrs_df['value'].str.replace(r"\W", " ")
    attrs_df['value'] = attrs_df['value'].str.replace(r"\s+", " ")
    attrs_df['value'] = attrs_df['value'].apply(lambda val: ' '.join([stemmer.stem(i) for i in make_string(val).split(" ") if i not in stop]))

    return attrs_df.groupby('product_uid')['value'].apply(lambda x: ' '.join(x.unique()))


# set of columns (processed and stemmed): title, description, brand, material, color, search_term
def extract_process_and_save_features1():
    start_time = datetime.datetime.now()

    train_data = pd.read_csv('./dataset/train.csv')
    test_data = pd.read_csv('./dataset/test.csv')
    numtrain = train_data.shape[0]
    data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    descr_by_prod = descr_by_product()
    attrs = pd.read_csv('./dataset/attributes.csv')
    brand_by_prod = extract_attribute_map(attrs, 'brand')
    color_by_prod = extract_attribute_map(attrs, 'color')
    material_by_prod = extract_attribute_map(attrs, 'material')

    data['descr'] = data.apply(lambda row: descr_by_prod[row['product_uid']], axis=1)
    data['brand'] = data.apply(lambda row: brand_by_prod[row['product_uid']] if row['product_uid'] in brand_by_prod else '', axis=1)
    data['color'] = data.apply(lambda row: color_by_prod[row['product_uid']] if row['product_uid'] in color_by_prod else '', axis=1)
    data['material'] = data.apply(lambda row: material_by_prod[row['product_uid']] if row['product_uid'] in material_by_prod else '', axis=1)

    data['product_title'] = data['product_title'].map(lambda x:process_str(x))
    data['descr'] = data['descr'].map(lambda x:process_str(x))
    data['search_term'] = data['search_term'].map(lambda x:process_str(x))

    new_train = data[:numtrain]
    new_test = data[numtrain:]
    new_test = new_test.drop('relevance', axis=1)

    new_train.to_csv('./dataset/train_new1.csv', index=None, encoding='utf-8')
    new_test.to_csv('./dataset/test_new1.csv', index=None, encoding='utf-8')

    print 'Processing time = ', (datetime.datetime.now() - start_time)


def count_common_words(str1, str2):
    if type(str1) is float or type(str2) is float:
        return 0

    set1 = set(str1.split(" "))
    set2 = set(str2.split(" "))
    return len(set1 & set2)


# Features:
#   - number of common words between search_term and product_title, product_description, brand, color, material
#   - length of search_term
#   - ratio of number of common words to length of search_term
def load_features1():
    train_new = pd.read_csv('./dataset/train_new1.csv', index_col='id')
    test_new = pd.read_csv('./dataset/test_new1.csv', index_col='id')

    numtrain = train_new.shape[0]
    data = pd.concat([train_new, test_new], axis=0)

    new_df = pd.DataFrame({'fake': '-'}, index=data.index)

    i = 0
    for index, row in data.iterrows():
        i += 1
        if i % 10000 == 0:
            print "%d rows of %d processed" % (i, data.shape[0])

        new_df.at[index, 'words_in_title'] = count_common_words(row['product_title'], row['search_term'])
        new_df.at[index, 'words_in_descr'] = count_common_words(row['descr'], row['search_term'])
        new_df.at[index, 'words_in_brand'] = count_common_words(row['brand'], row['search_term'])
        new_df.at[index, 'words_in_color'] = count_common_words(row['color'], row['search_term'])
        new_df.at[index, 'words_in_material'] = count_common_words(row['material'], row['search_term'])

    new_df['query_len'] = data['search_term'].map(lambda x: len(x.split(" ")))
    new_df['ratio_title'] = new_df['words_in_title']/new_df['query_len']
    new_df['ratio_descr'] = new_df['words_in_descr']/new_df['query_len']
    new_df['ratio_brand'] = new_df['words_in_brand']/new_df['query_len']
    new_df['ratio_color'] = new_df['words_in_color']/new_df['query_len']
    new_df['ratio_material'] = new_df['words_in_material']/new_df['query_len']

    new_df = new_df.drop('fake', axis=1)

    train_new_processed = new_df[:numtrain].copy()
    test_new_processed = new_df[numtrain:].copy()

    train_new_processed['relevance'] = train_new['relevance']

    return train_new_processed, test_new_processed