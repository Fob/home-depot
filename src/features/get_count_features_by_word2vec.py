import pandas as pd
import numpy as np
from gensim.models import Word2Vec

def get_count_features_by_word2vec(filename, threshold):

    data = pd.read_csv(filename, index_col='id')


    w2v_title_filename = "./src/features/prod_title_word2vec_model.model"
    w2v_title = Word2Vec.load(w2v_title_filename)
    w2v_descr_filename = "./src/features/descr_word2vec_model.model"
    w2v_descr = Word2Vec.load(w2v_descr_filename)


    features = pd.DataFrame({'words_in_title_w2v': 0, 'words_in_descr_w2v': 0, 'ratio_in_title_w2v': 0, 'ratio_in_descr_w2v': 0}, index=data.index)
    for n in data.index:
        if n % 1000 == 0:
            print "%d rows of %d processed" % (n, data.last_valid_index())
        row_n = data.loc[n]
        descr = set(row_n['descr'].split())
        title = set(row_n['product_title'].split())
        search = row_n['search_term'].split()

        title_count_value = 0
        descr_count_value = 0
        for term in search:
            #title
            if term in w2v_title.vocab:
                title_count_value = title_count_value + np.sum([w2v_title.similarity(term, part)>=threshold for part in title])
            #description
            if term in w2v_descr.vocab:
                descr_count_value = descr_count_value + np.sum([w2v_descr.similarity(term, part)>=threshold for part in descr])

        features.loc[n] = np.array([ np.double(descr_count_value)/len(search), np.double(title_count_value)/len(search), descr_count_value, title_count_value])

    print "Done!"

    return features