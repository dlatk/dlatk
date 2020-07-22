# cython: linetrace=True
# cython: language_level=3

import random
import sys

import numpy as np

from cython.view cimport array as cvarray
from timeit import default_timer as timer

class Document:
    
    def __init__(self, str doc_id, long[:] doc_tokens, long[:] doc_topics, long[:] topic_changes, long[:] doc_topic_counts):
        self.doc_id = doc_id
        self.doc_tokens = doc_tokens
        self.doc_topics = doc_topics
        self.topic_changes = topic_changes
        self.doc_topic_counts = doc_topic_counts

cdef class TopicModel:
    
    cdef long[:] topic_totals
    cdef long[:,:] word_topics
    cdef int num_topics
    cdef int vocab_size
    
    cdef double[:] topic_probs
    cdef double[:] topic_normalizers
    cdef float doc_smoothing
    cdef float word_smoothing
    cdef float smoothing_times_vocab_size
    
    documents = []
    vocabulary = []
    
    def __init__(self, num_topics, vocabulary, doc_smoothing, word_smoothing):
        self.num_topics = num_topics
        self.vocabulary.extend(vocabulary)
        self.vocab_size = len(vocabulary)
        
        self.doc_smoothing = doc_smoothing
        self.word_smoothing = word_smoothing
        self.smoothing_times_vocab_size = word_smoothing * self.vocab_size
        
        self.topic_totals = np.zeros(num_topics, dtype=int)
        self.word_topics = np.zeros((self.vocab_size, num_topics), dtype=int)
    
    def add_document(self, doc):
        cdef int word_id, topic
        
        self.documents.append(doc)
        
        for i in range(len(doc.doc_tokens)):
            word_id = doc.doc_tokens[i]
            topic = doc.doc_topics[i]
            
            self.word_topics[word_id,topic] += 1
            self.topic_totals[topic] += 1
            doc.doc_topic_counts[topic] += 1
            
    def sample(self, iterations):
        cdef int old_topic, new_topic, word_id, topic, i, doc_length
        cdef double sampling_sum = 0
        cdef double sample
        cdef long[:] word_topic_counts
        
        cdef long[:] doc_tokens
        cdef long[:] doc_topics
        cdef long[:] doc_topic_counts
        cdef long[:] topic_changes
        
        cdef double[:] uniform_variates
        cdef double[:] topic_probs = np.zeros(self.num_topics, dtype=float)
        cdef double[:] topic_normalizers = np.zeros(self.num_topics, dtype=float)
        
        for topic in range(self.num_topics):
            topic_normalizers[topic] = 1.0 / (self.topic_totals[topic] + self.smoothing_times_vocab_size)
        
        
        for iteration in range(iterations):
            for document in self.documents:
                doc_tokens = document.doc_tokens
                doc_topics = document.doc_topics
                doc_topic_counts = document.doc_topic_counts
                topic_changes = document.topic_changes
                
                doc_length = len(document.doc_tokens)
                uniform_variates = np.random.random_sample(doc_length)
                
                for i in range(doc_length):
                    word_id = doc_tokens[i]
                    old_topic = doc_topics[i]
                    word_topic_counts = self.word_topics[word_id,:]
        
                    ## erase the effect of this token
                    word_topic_counts[old_topic] -= 1
                    self.topic_totals[old_topic] -= 1
                    doc_topic_counts[old_topic] -= 1
        
                    topic_normalizers[old_topic] = 1.0 / (self.topic_totals[old_topic] + self.smoothing_times_vocab_size)
        
                    ###
                    ### SAMPLING DISTRIBUTION
                    ###
        
                    sampling_sum = 0.0
                    for topic in range(self.num_topics):
                        topic_probs[topic] = (doc_topic_counts[topic] + self.doc_smoothing) * (word_topic_counts[topic] + self.word_smoothing) * topic_normalizers[topic]
                        sampling_sum += topic_probs[topic]
        
                    #sample = random.uniform(0, sampling_sum)
                    #sample = np.random.random_sample() * sampling_sum
                    sample = uniform_variates[i] * sampling_sum
        
                    new_topic = 0
                    while sample > topic_probs[new_topic]:
                        sample -= topic_probs[new_topic]
                        new_topic += 1
            
                    ## add back in the effect of this token
                    word_topic_counts[new_topic] += 1
                    self.topic_totals[new_topic] += 1
                    doc_topic_counts[new_topic] += 1
                    topic_normalizers[new_topic] = 1.0 / (self.topic_totals[new_topic] + self.smoothing_times_vocab_size)

                    doc_topics[i] = new_topic
        
                    if new_topic != old_topic:
                        #pass
                        topic_changes[i] += 1

    def print_topic(self, int topic):
        sorted_words = sorted(zip(self.word_topics[:,topic], self.vocabulary), reverse=True)

        for i in range(20):
            w = sorted_words[i]
            print("{}\t{}".format(w[0], w[1]))

    def print_all_topics(self, out=sys.stdout):
        for topic in range(self.num_topics):
            sorted_words = sorted(zip(self.word_topics[:,topic], self.vocabulary), reverse=True)
            print(str(topic), end="\t", file=out)
            print(str(self.doc_smoothing), end="\t", file=out)
            print(" ".join([w for x, w in sorted_words[:20]]), file=out)



def sample_doc(long[:] doc_tokens, long[:] doc_topics, long[:] topic_changes, long[:] doc_topic_counts, long[:,:] word_topics, long[:] topic_totals, double[:] topic_probs, double[:] topic_normalizers, float doc_smoothing, float word_smoothing, float smoothing_times_vocab_size, int num_topics):
    
    cdef int old_topic, new_topic, word_id, topic, i
    cdef double sampling_sum = 0
    cdef double sample
    cdef long[:] word_topic_counts
    
    cdef int doc_length = len(doc_tokens)
    cdef double[:] uniform_variates = np.random.random_sample(doc_length)
    
    for i in range(doc_length):
        word_id = doc_tokens[i]
        old_topic = doc_topics[i]
        word_topic_counts = word_topics[word_id,:]
        
        ## erase the effect of this token
        word_topic_counts[old_topic] -= 1
        topic_totals[old_topic] -= 1
        doc_topic_counts[old_topic] -= 1
        
        topic_normalizers[old_topic] = 1.0 / (topic_totals[old_topic] + smoothing_times_vocab_size)
        
        ###
        ### SAMPLING DISTRIBUTION
        ###
        
        sampling_sum = 0.0
        for topic in range(num_topics):
            topic_probs[topic] = (doc_topic_counts[topic] + doc_smoothing) * (word_topic_counts[topic] + word_smoothing) * topic_normalizers[topic]
            sampling_sum += topic_probs[topic]
        
        #sample = random.uniform(0, sampling_sum)
        #sample = np.random.random_sample() * sampling_sum
        sample = uniform_variates[i] * sampling_sum
        
        new_topic = 0
        while sample > topic_probs[new_topic]:
            sample -= topic_probs[new_topic]
            new_topic += 1
            
        ## add back in the effect of this token
        word_topic_counts[new_topic] += 1
        topic_totals[new_topic] += 1
        doc_topic_counts[new_topic] += 1
        topic_normalizers[new_topic] = 1.0 / (topic_totals[new_topic] + smoothing_times_vocab_size)

        doc_topics[i] = new_topic
        
        if new_topic != old_topic:
            #pass
            topic_changes[i] += 1
