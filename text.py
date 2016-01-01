from gensim import corpora, models, similarities
import os
import string
import lvq
import time

print "Creating topics dict..."
topics = {}
for mdir in os.listdir('lvq'):
    topics[mdir] = []
    for (i, mfile) in enumerate(os.listdir('lvq/' + mdir)):
        topics[mdir].append(open('lvq/' + mdir + '/' + mfile).read())

print "Translating documents..."
texts = []
table = string.maketrans("\n", " ")
for key in topics:
    documents = topics[key]
    for i in range(0, len(documents)):
        documents[i] = documents[i].translate(table, string.punctuation).lower()
    texts += [[word for word in document.split()]
        for document in documents]

print "Creating dictionary...",
dictionary = corpora.Dictionary(texts)
print "length", len(dictionary)

documents = [topics[key][i] for i in range(0,len(topics[key])) for key in topics]
classes = [i for doc in topics[key] for (i, key) in enumerate(topics)]
print "Computing vectors..."
vectors = [lvq.doc2vec(dictionary, document) for document in documents]
print "Creating LVQ..."
lvqn = lvq.lvq_net.from_dict(dictionary, len(vectors), 3)
print "Feeding vectors..."
lvqn.feed_multiple(zip(vectors, classes), len(vectors))
print "Done."
def check():
	right = 0
	wrong = 0
	for (vector, cl) in zip(vectors, classes):
		if lvqn.classify(vector) == cl:
			right += 1
		else:
			wrong += 1
	print right, "right,", wrong, "wrong."
while True:
	lvqn.feed_multiple(zip(vectors, classes), len(vectors))
	lvq.step -= (lvq.step - 0.1) * 0.1
	print "Step is", lvq.step
	print "Sleeping 5 sec..."
	time.sleep(5)
  
