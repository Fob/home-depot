import gensim
from gensim.models.doc2vec import TaggedLineDocument, LabeledSentence
from collections import namedtuple
import multiprocessing
from gensim.models import Doc2Vec, Word2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import pandas as pd
import re

#--------------------------
#LOGGING
#--------------------------
import logging
import os.path
import sys
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
#--------------------------

#text filtration
data = pd.DataFrame(pd.read_csv('./dataset/train.csv'))
test = data['product_title'][:10]
test = [re.sub("[^A-Za-z-\s\d/,\.]", "", row_string.lower()) for row_string in test]
test = [re.sub("\s(ft|in|cu|lb)\.", " \\1", row_string) for row_string in test]
[re.sub("\s([A-Za-z]+)(\.|,)", " \\1", row_string) for row_string in test]


product_data = pd.DataFrame(pd.read_csv('./dataset/product_descriptions.csv'))
all_text_data = product_data['product_description'].iloc[:1000]
sentences = [row.lower().split() for row in all_text_data]
cores = multiprocessing.cpu_count()
model = Word2Vec(sentences, min_count=1, workers=cores, alpha=0.025, min_alpha=0.025)
model.most_similar('angle')

for n in range(10):
    print n
    model.train(sentences=sentences)

model.most_similar('hammer')


train_data = pd.DataFrame(pd.read_csv('./dataset/train.csv')[0:5])


for n in range(len(train_data)):
    row_n = train_data.iloc[n]
    prod_title = row_n['product_title'].lower().split()
    prod_uid = row_n['product_uid']
    search_term = row_n['search_term'].lower().split()
    product_desc = product_data[product_data['product_uid'] == prod_uid]['product_description'].iloc[0].lower().split()

    #value_prod_title = model.n_similarity(search_term, prod_title)
    value_prod_desc = model.n_similarity(search_term, product_desc)

    print n
    #print value_prod_title
    print search_term
    print prod_title
    print value_prod_desc
    print '---------------'

#train_data = pd.DataFrame(pd.read_csv('./dataset/train.csv'))
#train_data['relevance'].unique()

