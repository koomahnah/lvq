from gensim import corpora, models, similarities
import os
import string
import lvq

topics = {}
for mdir in os.listdir('lvq'):
    topics[mdir] = []
    for mfile in os.listdir('lvq/' + mdir):
        topics[mdir].append(open('lvq/' + mdir + '/' + mfile).read())

print topics['science'][0]

texts = []
table = string.maketrans("\n", " ")
for key in topics:
    documents = topics[key]
    for i in range(0, len(documents)):
        documents[i] = documents[i].translate(table, string.punctuation).lower()
    texts += [[word for word in document.split()]
        for document in documents]

dictionary = corpora.Dictionary(texts)
print len(dictionary)

lvqn = lvq.lvq_net.from_dict(dictionary, 500, 3)
for topic_index, key in enumerate(topics):
    print 'Topic ', key
    documents = topics[key]
    for doc_index, document in enumerate(documents):
        print 'Feeding document ', doc_index
        vec = lvq.doc2vec(dictionary, document)
        lvqn.feed_until(vec, topic_index, 1)
