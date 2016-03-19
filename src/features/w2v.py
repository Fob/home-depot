import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from numpy import linalg as la

def get_features_cos_between_sum_of_vectors(filename):

    data = pd.read_csv(filename, index_col='id')

    w2v_filename = "./src/features/word2vec_model_whole.model"
    w2v_model = Word2Vec.load(w2v_filename)

    features = pd.DataFrame({'sim_with_title_w2v': 0, 'sim_with_descr_w2v': 0}, index=data.index)
    for n in data.index:
        if n % 1000 == 0:
            print "%d rows of %d processed" % (n, data.last_valid_index())
        row_n = data.loc[n]
        descr = set(row_n['descr'].split())
        title = set(row_n['product_title'].split())
        search = row_n['search_term'].split()


        #search vector
        if np.sum([not term in (w2v_model.vocab) for term in search]) != 0:
            sim_with_title_w2v = 0
            sim_with_descr_w2v = 0
        else:
            search_vector = np.sum(w2v_model[search], axis=0)/len(search)
            #title_vector
            if np.sum([not term in (w2v_model.vocab) for term in title]) != 0:
                sim_with_title_w2v = 0
            else:
                title_vector = np.sum(w2v_model[title], axis=0)/len(title)
                #descr_vector
                if np.sum([not term in (w2v_model.vocab) for term in descr]) != 0:
                    sim_with_descr_w2v = 0
                else:
                    descr_vector = np.sum(w2v_model[descr], axis=0)/len(descr)
                    sim_with_title_w2v = np.dot(search_vector, title_vector)/(la.norm(search_vector)*la.norm(title_vector))
                    sim_with_descr_w2v = np.dot(search_vector, descr_vector)/(la.norm(search_vector)*la.norm(descr_vector))

        features.loc[n] = np.array([sim_with_descr_w2v, sim_with_title_w2v])

    print "Done!"

    return features
