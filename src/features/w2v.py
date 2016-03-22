import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from numpy import linalg as la

def cos_sim(a,b):
    return np.dot(a, b)/(la.norm(a)*la.norm(b))


def get_features_sum_of_vects_as_doc_and_load_model(filename):

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

def read_w2v_model_from_text_file(filename):
    w2v = {}
    n=0
    num_lines = sum(1 for line in open(filename))
    print "Counted Number of Lines: " + str(num_lines)
    with open(filename) as f:
        for line in f:
            tokens = line.split()
            w2v[tokens[0]] = np.array(map(float, tokens[1:]))
            n=n+1
            if n%10000 == 0:
                print num_lines-n
    print "Finished"

    return w2v


def get_features_sum_of_vects_as_doc(w2v_model, data):

    features = pd.DataFrame({'sim_with_title_w2v': 0, 'sim_with_descr_w2v': 0}, index=data.index)
    for n in data.index:
        if n % 1000 == 0:
            print "%d rows of %d processed" % (n, data.last_valid_index())
        row_n = data.loc[n]
        descr = set(row_n['descr'].split())
        title = set(row_n['product_title'].split())
        search = row_n['search_term'].split()


        #search vector
        if np.sum([not term in (w2v_model) for term in search]) != 0:
            sim_with_title_w2v = 0
            sim_with_descr_w2v = 0
        else:
            search_vector = sum([w2v_model[term] for term in search])/len(search)
            #title_vector
            if np.sum([not term in (w2v_model) for term in title]) != 0:
                sim_with_title_w2v = 0
            else:
                title_vector = sum([w2v_model[term] for term in title])/len(title)
                sim_with_title_w2v = np.dot(search_vector, title_vector)/(la.norm(search_vector)*la.norm(title_vector))
                #descr_vector
            if np.sum([not term in (w2v_model) for term in descr]) != 0:
                sim_with_descr_w2v = 0
            else:
                descr_vector = sum([w2v_model[term] for term in descr])/len(descr)
                sim_with_descr_w2v = np.dot(search_vector, descr_vector)/(la.norm(search_vector)*la.norm(descr_vector))


        features.loc[n] = np.array([sim_with_descr_w2v, sim_with_title_w2v])

    print "Done!"

    return features




def get_features_count_synonyms(w2v_model, data, threshold):


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
        for search_term in search:
            #title
            if search_term in w2v_model:
                title_count_value = title_count_value + sum([cos_sim(w2v_model[search_term], w2v_model[term])>threshold for term in title if ((term in w2v_model) & (term!=search_term))])
            #description
            if search_term in w2v_model:
                descr_count_value = descr_count_value + sum([cos_sim(w2v_model[search_term], w2v_model[term])>threshold for term in descr if ((term in w2v_model) & (term!=search_term))])

        features.loc[n] = np.array([ np.double(descr_count_value)/len(search), np.double(title_count_value)/len(search), descr_count_value, title_count_value])

    print "Done!"

    return features