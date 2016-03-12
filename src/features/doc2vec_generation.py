import gensim
from gensim.models.doc2vec import TaggedLineDocument, LabeledSentence
from collections import namedtuple
import multiprocessing
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict


cores = multiprocessing.cpu_count()
#LabeledSentence(['what', 'happens', 'when', 'an', 'army', 'of', 'wetbacks', 'towelheads', 'and', 'godless', 'eastern', 'european', 'commies', 'gather', 'their', 'forces', 'south', 'of', 'the', 'border', 'gary', 'busey', 'kicks', 'their', 'butts', 'of', 'course', 'another', 'laughable', 'example', 'of', 'reagan-era', 'cultural', 'fallout', 'bulletproof', 'wastes', 'a', 'decent', 'supporting', 'cast', 'headed', 'by', 'l', 'q', 'jones', 'and', 'thalmus', 'rasulala'], ['LABELED_10', '0'])`
documents = TaggedLineDocument('./dataset/test_doc2vec.txt')
doc2vec_model = Doc2Vec(alpha=0.025, min_alpha=0.025)
#doc2vec_model = Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores)
doc2vec_model.build_vocab(documents)

LabeledSentence(words='How are you'.split())



sentences = [['first', 'hello', 'man'], ['second', 'hi', 'woman']]
model = gensim.models.Doc2Vec(sentences, min_count=1)
model.similarity('woman', 'man')




doc1=["This is a sentence","This is another sentence"]
documents1=[doc.strip().split(" ") for doc in doc1 ]
documents = models.doc2vec.LabeledSentence(words=[u'some', u'words', u'here'], labels=[u'SENT_1'])
model = models.Doc2Vec(documents, size = 100, window = 300, min_count = 10, workers=4)

