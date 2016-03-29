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
        if n % 10000 == 0:
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




def get_features_sum_of_vects_as_doc_fast_method(w2v_model, data):

    def get_cosine_value(search, title, descr, n):

        if n % 10000 == 0:
            print "%d rows processed" % (n)

        search = search.split()
        title = title.split()
        descr = descr.split()

        #search vector
        if sum([term in (shared_w2v_model) for term in search]) == 0:
            sim_with_title_w2v = 0
            sim_with_descr_w2v = 0
        else:
            vectors = [shared_w2v_model[term] for term in search if term in shared_w2v_model]
            search_vector = sum(vectors, 0)/len(vectors)
            #search_vector = sum([w2v_model[term] for term in search])/len(search)

            #title_vector
            if sum([term in (shared_w2v_model) for term in title]) == 0:
                sim_with_title_w2v = 0
            else:
                vectors = [shared_w2v_model[term] for term in title if term in shared_w2v_model]
                title_vector = sum(vectors, 0)/len(vectors)
                sim_with_title_w2v = np.dot(search_vector, title_vector)/(la.norm(search_vector)*la.norm(title_vector))

            #descr_vector
            if sum([term in (shared_w2v_model) for term in descr]) == 0:
                sim_with_descr_w2v = 0
            else:
                vectors = [shared_w2v_model[term] for term in descr if term in shared_w2v_model]
                descr_vector = sum(vectors, 0)/len(vectors)
                sim_with_descr_w2v = np.dot(search_vector, descr_vector)/(la.norm(search_vector)*la.norm(descr_vector))


        return np.array([sim_with_title_w2v, sim_with_descr_w2v])

    global shared_w2v_model
    shared_w2v_model = w2v_model
    result = map(lambda search, title, descr, n: get_cosine_value(search, title, descr, n), data['search_term'], data['product_title'], data['descr'], range(0,len(data)))
    features = pd.DataFrame(result, columns=['sim_with_title_w2v', 'sim_with_descr_w2v'], index=data.index)
    return features



def get_features_count_synonyms(w2v_model, data, threshold):

    def get_cosine_value(search, title, descr, n):

        if n % 1000 == 0:
            print "%d rows processed" % (n)

        descr = descr.split()
        title = title.split()
        search = search.split()

        title_count_value = 0
        descr_count_value = 0
        for search_term in search:
            #title
            if search_term in shared_w2v_model:
                title_count_value = title_count_value + sum([cos_sim(shared_w2v_model[search_term], shared_w2v_model[term])>threshold for term in title if ((term!=search_term) & (term in shared_w2v_model))])
            #description
            if search_term in shared_w2v_model:
                descr_count_value = descr_count_value + sum([cos_sim(shared_w2v_model[search_term], shared_w2v_model[term])>threshold for term in descr if ((term!=search_term) & (term in shared_w2v_model))])

        return np.array([title_count_value, np.double(title_count_value)/len(search), descr_count_value, np.double(descr_count_value)/len(search)])

    global shared_w2v_model
    shared_w2v_model = w2v_model
    result = map(lambda search, title, descr, n: get_cosine_value(search, title, descr, n), data['search_term'], data['product_title'], data['descr'], range(0, len(data)))
    features = pd.DataFrame(result, columns=['words_in_title_w2v', 'ratio_in_title_w2v', 'words_in_descr_w2v', 'ratio_in_descr_w2v'], index=data.index)

    print "Done!"
    return features




def get_count_of_words(data):


    def get_cnt_value(title, descr, n):
        if n % 10000 == 0:
            print "%d rows processed" % (n)
        descr = descr.split()
        title = title.split()
        return np.array([ len(title), len(descr)])

    result = map(lambda title, descr, n: get_cnt_value(title, descr, n), data['product_title'], data['descr'], range(0,len(data)))
    features = pd.DataFrame(result, columns=['cnt_in_title', 'cnt_in_descr'], index=data.index)

    print "Done!"
    return features



