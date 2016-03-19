import gensim
import numpy as np
from gensim.models.doc2vec import TaggedLineDocument, LabeledSentence
from collections import namedtuple
import multiprocessing as mlt
from gensim.models import Doc2Vec, Word2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import pandas as pd
import re
from scipy.sparse import hstack, vstack

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
all_prod_title = data['product_title'][:10000]
all_prod_title = [re.sub("[^A-Za-z-\s\d/,\.]", "", row_string.lower()) for row_string in all_prod_title]
all_prod_title = [re.sub("\s(ft|in|cu|lb)\.", " \\1", row_string) for row_string in all_prod_title]
all_prod_title = [re.sub("\s([A-Za-z]+)(\.|,)", " \\1", row_string) for row_string in all_prod_title]
print "filtered"

#get filtered data
train_data = pd.DataFrame(pd.read_csv('./dataset/train_new1.csv'))
test_data = pd.DataFrame(pd.read_csv('./dataset/test_new1.csv'))
cores = mlt.cpu_count()






#PRODUCT TITLE
#----------------------------------------------------------------------------------
train_prod_title = train_data['product_title']
train_prod_title_sentences = [row.split() for row in train_prod_title]
test_prod_title = test_data['product_title']
test_prod_title_sentences = [row.split() for row in test_prod_title]
prod_title_sentences = train_prod_title_sentences + test_prod_title_sentences
model = Word2Vec(prod_title_sentences, min_count=1, workers=cores, alpha=0.025, min_alpha=0.025)
for n in range(50):
    print n
    model.train(sentences=prod_title_sentences)

model.most_similar('bosch')
model.most_similar('makita')
model.most_similar('drill')

prod_title_model_filename = "./src/features/prod_title_word2vec_model.model"
title_loaded_model.save(prod_title_model_filename)

#train model
train_prod_title = train_data['product_title']
train_prod_title_sentences = [row.split() for row in train_prod_title]
test_prod_title = test_data['product_title']
test_prod_title_sentences = [row.split() for row in test_prod_title]
prod_title_sentences = train_prod_title_sentences + test_prod_title_sentences
prod_title_model_filename = "./src/features/prod_title_word2vec_model.model"
title_loaded_model = Word2Vec.load(prod_title_model_filename)
for n in range(50):
    print n
    title_loaded_model.train(sentences=prod_title_sentences)
title_loaded_model.save(prod_title_model_filename)


# generate features
title_loaded_model = Word2Vec.load(prod_title_model_filename)
sim_value_prod_title = []
for n in range(len(train_data)):
    row_n = train_data.iloc[n]
    prod_title = row_n['product_title'].split()
    search_term = row_n['search_term'].split()
    if all(term in title_loaded_model.vocab for term in search_term):
        sim_value_prod_title = np.append(sim_value_prod_title, title_loaded_model.n_similarity(search_term, prod_title))
    else:
        sim_value_prod_title = np.append(sim_value_prod_title, 0)

    if n%10000 == 0:
        print len(train_data) - n
#    print n
#    print search_term
#    print prod_title
#    print value_prod_desc
#    print '---------------'

print sim_value_prod_title.shape

#----------------------------------------------------------------------------------








#DESCRIPTION
#----------------------------------------------------------------------------------
train_descr = train_data['descr']
train_descr_sentences = [row.split() for row in train_descr]
test_descr = test_data['descr']
test_descr_sentences = [row.split() for row in test_descr]
descr_sentences = train_descr_sentences + test_descr_sentences
descr_model = Word2Vec(descr_sentences, min_count=1, workers=cores, alpha=0.025, min_alpha=0.025)
for n in range(50):
    print n
    descr_model.train(sentences=descr_sentences)
descr_model_filename = "./src/features/descr_word2vec_model.model"
descr_model.save(descr_model_filename)

descr_model.most_similar('bosch')
descr_model.most_similar('makita')
descr_model.most_similar('drill')

descr_model_filename = "./src/features/descr_word2vec_model.model"
descr_loaded_model = Word2Vec.load(descr_model_filename)
descr_model = descr_loaded_model
#descr_model.save(descr_model_filename)

#train model
train_descr = train_data['descr']
train_descr_sentences = [row.split() for row in train_descr]
test_descr = test_data['descr']
test_descr_sentences = [row.split() for row in test_descr]
descr_sentences = train_descr_sentences + test_descr_sentences
descr_model_filename = "./src/features/descr_word2vec_model.model"
descr_loaded_model = Word2Vec.load(descr_model_filename)
for n in range(50):
    print n
    descr_loaded_model.train(sentences=descr_sentences)
descr_loaded_model.save(descr_model_filename)
descr_loaded_model.most_similar('bracket')


# generate features
descr_loaded_model = Word2Vec.load(descr_model_filename)
sim_value_prod_descr = []
for n in range(len(train_data)):
    row_n = train_data.iloc[n]
    descr = row_n['descr'].split()
    search_term = row_n['search_term'].split()
    if all(term in descr_loaded_model.vocab for term in search_term):
        sim_value_prod_descr = np.append(sim_value_prod_descr, descr_loaded_model.n_similarity(search_term, descr))
    else:
        sim_value_prod_descr = np.append(sim_value_prod_descr, 0)

    if n%10000 == 0:
        print len(train_data) - n
#    print n
#    print search_term
#    print prod_title
#    print value_prod_desc
#    print '---------------'

print sim_value_prod_descr.shape













#----------------------------------------------------------------------------------
# TEST
# SAVE NEW FEATURES FOR TEST TO FILE

prod_title_model_filename = "./src/features/prod_title_word2vec_model.model"
title_loaded_model = Word2Vec.load(prod_title_model_filename)
descr_model_filename = "./src/features/descr_word2vec_model.model"
descr_loaded_model = Word2Vec.load(descr_model_filename)

current_data = train_data

sim_value_prod_descr = []
sim_value_prod_title = []
for n in range(len(current_data)):
    row_n = current_data.iloc[n]
    descr = row_n['descr'].split()
    prod_title = row_n['product_title'].split()
    search_term = row_n['search_term'].split()

    descr_sim_value = 0
    title_sim_value = 0
    for term in search_term:
        sim_descr_sum = 0
        sim_title_sum = 0
        #descr
        if term in descr_loaded_model.vocab:
            for descr_part in descr:
                sim_descr_sum  = sim_descr_sum  + descr_loaded_model.similarity(term, descr_part)
        descr_sim_value = descr_sim_value + sim_descr_sum
        #title
        if term in title_loaded_model.vocab:
            for title_part in prod_title:
                sim_title_sum  = sim_title_sum  + title_loaded_model.similarity(term, title_part)
        title_sim_value = title_sim_value + sim_title_sum

    descr_sim_value = descr_sim_value/len(search_term)
    title_sim_value = descr_sim_value/len(search_term)

    sim_value_prod_descr = np.append(sim_value_prod_descr, descr_sim_value)
    sim_value_prod_title = np.append(sim_value_prod_title, title_sim_value)

    #title
#    if all(term in title_loaded_model.vocab for term in search_term):
#        sim_value_prod_title = np.append(sim_value_prod_title, title_loaded_model.n_similarity(search_term, prod_title))
#    else:
#        sim_value_prod_title = np.append(sim_value_prod_title, 0)
    #descr
#    if all(term in descr_loaded_model.vocab for term in search_term):
#        sim_value_prod_descr = np.append(sim_value_prod_descr, descr_loaded_model.n_similarity(search_term, descr))
#    else:
#        sim_value_prod_descr = np.append(sim_value_prod_descr, 0)

    if n%1000 == 0:
        print len(current_data) - n


#----------------------------------------------------------------------------------
# TRAIN
# SAVE NEW FEATURES FOR TRAIN TO FILE

new_features_train = pd.DataFrame({'similarity_search_title': sim_value_prod_title, 'similarity_search_prod_descr': sim_value_prod_descr})
new_features_train.to_csv('./src/features/word2vec_features_bywords_train.csv', index=None)


#----------------------------------------------------------------------------------
# TEST
# SAVE NEW FEATURES FOR TEST TO FILE

new_features_test = pd.DataFrame({'similarity_search_title': sim_value_prod_title, 'similarity_search_prod_descr': sim_value_prod_descr})
new_features_test.to_csv('./src/features/word2vec_features_bywords_test.csv', index=None)


#----------------------------------------------------------------------------------
# Test getSimilarWords

import src.features.getSimilarWords as sim
sim.getSimilarWords('makita', 'description', 0.5)

#----------------------------------------------------------------------------------
#TRAIN WHOLE MODEL: TITLE+DESCRIPTION
#----------------------------------------------------------------------------------
train_prod_title = train_data['product_title']
train_prod_title_sentences = [row.split() for row in train_prod_title]
test_prod_title = test_data['product_title']
test_prod_title_sentences = [row.split() for row in test_prod_title]
prod_title_sentences = train_prod_title_sentences + test_prod_title_sentences
descr_model_filename = "./src/features/descr_word2vec_model.model"
descr_loaded_model = Word2Vec.load(descr_model_filename)
for n in range(5):
    print n
    descr_loaded_model.train(sentences=prod_title_sentences, min_count=1)
whole_model_filename = "./src/features/word2vec_model_whole.model"
descr_loaded_model.save(whole_model_filename)

(descr_loaded_model['makita']+descr_loaded_model['bosch'])/2


#----------------------------------------------------------------------------------


