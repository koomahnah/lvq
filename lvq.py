import random
import math
import Image
from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial

attract_step = 0.9
repel_step = 0.5
bias_param = 1
bias_base = 2
def load_image(name):
    im = Image.open(name)
    return list(im.getdata())

def print_image(image, size):
    for i in range(0,size):
        print image[i],
        if i == 0:
            continue
        elif (i % math.sqrt(size)) == (math.sqrt(size) - 1):
            print

def doc2vec(dictionary, document):
    text = [word for word in document.split()]
    occurences = dictionary.doc2bow(text)
    vec = [0] * len(dictionary) 
    for index, count in occurences:
        vec[index] = 1
    return vec

class lvq_neuron:
    def __init__(self, weights=None, input_dim=1, assigned_class=0):
        if weights is None:
            self.weights = [0] * input_dim
            for i in range(0, input_dim):
                self.weights[i] = random.gauss(0,0.5)
        else:
            self.weights = weights
        self.input_dim = input_dim
        self.assigned_class = assigned_class
        self.bias = 0.0
    def distance(self, vec):
        if len(vec) != self.input_dim:
            raise Exception("wrong length of input vector")
        sum = 0
        for i in range(0, self.input_dim):
            sum += math.pow(self.weights[i] - vec[i], 2)
        distance = math.sqrt(sum)
        return distance
    def score(self, vec):
        if len(vec) != self.input_dim:
            raise Exception("wrong length of input vector")
        return self.distance(vec) * (bias_base ** self.bias)
    def attract(self, vec):
        if len(vec) != self.input_dim:
            raise Exception("wrong length of input vector")
        delta = [0] * self.input_dim
        for i in range(0, self.input_dim):
            delta[i] = attract_step * (vec[i] - self.weights[i])
            self.weights[i] = self.weights[i] + delta[i]
        print "attract, new distance is", self.distance(vec)
    def repel(self, vec):
        if len(vec) != self.input_dim:
            raise Exception("wrong length of input vector")
        delta = [0] * self.input_dim
        for i in range(0, self.input_dim):
            delta[i] = repel_step * (vec[i] - self.weights[i]) * (-1)
            self.weights[i] = self.weights[i] + delta[i]
        print "repel, new distance is", self.distance(vec)

def comp_score(x, y, dim):
    sum = 0
    for i in range(0, dim):
        sum += math.pow(x[i] - y[i], 2)
    distance = math.sqrt(sum)
    return distance

class lvq_net:
    def __init__(self, neuron_count, input_dim, output_dim):
        self.neurons = [0] * neuron_count
        for i in range(0, neuron_count):
            self.neurons[i] = lvq_neuron(input_dim=input_dim,
                assigned_class=(i * output_dim / neuron_count))
            print "assigned class", i * output_dim / neuron_count, "to neuron", i
        self.neuron_count = neuron_count
        self.input_dim = input_dim
        self.output_dim = output_dim
    @ classmethod
    def from_dict(cls, dictionary, neuron_count, output_dim):
        return cls(neuron_count, len(dictionary), output_dim)
    def compete(self, vec, count_bias=True):
        if len(vec) != self.input_dim:
            raise Exception("wrong length of input vector")
        min_index = 0
        min_score = 0
        scores = [0] * self.neuron_count
        if count_bias:
            for i, neuron in enumerate(self.neurons):
                scores[i] = neuron.score(vec)
        else:
            for i, neuron in enumerate(self.neurons):
                scores[i] = neuron.distance(vec)
#        scores = Parallel(n_jobs=4)(delayed(comp_score)(self.neurons[i], vec,
#                    count_tiredness) for i in range(self.neuron_count))
#        f = partial(comp_score, y=vec, dim=self.input_dim)
#        pool = Pool(processes=2, maxtasksperchild=1)
#        scores = pool.map(f, [neuron.weights for neuron in self.neurons])

        for i in range(0, self.neuron_count):
            if scores[i] < min_score or i == 0:
                min_index = i
                min_score = scores[i]
#        for i, e in enumerate(scores):
#            if e < min_score or i == 0:
#                min_index = i
#                min_score = e 

        print "winner is", min_index, "with score", min_score,
        print "(bias", self.neurons[min_index].bias, ")"
        return min_index, min_score
    def feed(self, vec, desired_class):
        winner_index, winner_score = self.compete(vec)
        winner = self.neurons[winner_index]
        if winner.assigned_class == desired_class:
            winner.attract(vec)
            winner.bias += bias_param
        else:
            winner.repel(vec)
#        for neuron in self.neurons:
#            if neuron is not winner:
#                neuron.bias -= (neuron.bias - 1) * 0.1
    def feed_multiple(self, vec_class, iterations):
        done = [False] * len(vec_class)
        print "Feeding class of", len(vec_class), "vectors",
        for i in range(iterations):
            print "Cycle", i, "out of", iterations, "...",
            index = random.randint(0, len(vec_class) - 1)
            print "feeding vector", index
            self.feed(vec_class[index][0], vec_class[index][1])

    def feed_until(self, vec, desired_class, threshold):
        """ Feed until distance is below threshold. """
        winner_index, score = self.compete(vec)
        winner = self.neurons[winner_index]
        last_attracted = False
        while score > threshold:
            if winner.assigned_class == desired_class or last_attracted:
                winner.attract(vec)
                last_attracted = True
                score = winner.score(vec)
                winner.tiredness += 1
            else:
                winner.repel(vec)
                winner_index, score = self.compete(vec)
                winner = self.neurons[winner_index]

    def feed_doc(self, dictionary, document, desired_class):
        self.feed(doc2vec(dictionary, document), desired_class)
    def classify_doc(self, dictionary, document):
        self.classify(doc2vec(dictionary, document))
    def classify(self, vec):
	index, score = self.compete(vec, False)
        return self.neurons[index].assigned_class
    def show_neuron(self, neuron_index):
        neuron = self.neurons[neuron_index]
        ret = [0] * self.input_dim
        for i in range(0, self.input_dim):
            ret[i] = int(round(neuron.weights[i]))
        print_image(ret,self.input_dim)
        return ret
