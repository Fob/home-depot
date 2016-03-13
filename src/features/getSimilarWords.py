from gensim.models import Word2Vec

def getSimilarWords(word, field, threshold):

    if field == 'title':
        w2v_model_filename = "./src/features/prod_title_word2vec_model.model"
    elif field == 'description':
        w2v_model_filename = "./src/features/descr_word2vec_model.model"
    w2v_model = Word2Vec.load(w2v_model_filename)

    if word in w2v_model.vocab:
        similar_words = w2v_model.most_similar(word, topn=100)
        similar_words_by_threshold = [a for i, (a,b) in enumerate(similar_words) if b>=threshold]
    else:
        similar_words_by_threshold = []

    return similar_words_by_threshold

