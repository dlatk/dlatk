#!/usr/bin/env python
#########################################
import re

import csv
import os, sys

from dlatk.messageTransformer import MessageTransformer

sys.path.append(os.path.dirname(os.path.realpath(__file__)).replace("/dlatk/LexicaInterface",""))

from dlatk.featureExtractor import FeatureExtractor
from dlatk import dlaConstants as dlac

from gensim import corpora
from gensim.models.wrappers import LdaMallet

from numpy import log2, isnan
from pymallet import defaults
from pymallet.lda import estimate_topics

from json import loads

class TopicExtractor(FeatureExtractor):

    def __init__(self, corpdb=dlac.DEF_CORPDB, corptable=dlac.DEF_CORPTABLE, correl_field=dlac.DEF_CORREL_FIELD,
                 mysql_host = "localhost", message_field=dlac.DEF_MESSAGE_FIELD, messageid_field=dlac.DEF_MESSAGEID_FIELD,
                 encoding=dlac.DEF_ENCODING, use_unicode=dlac.DEF_UNICODE_SWITCH, ldaMsgTable =dlac.DEF_LDA_MSG_TABLE):
        super(TopicExtractor, self).__init__(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode)
        self.ldaMsgTable = ldaMsgTable


    def createDistributions(self, filename=None):

        if not filename:
            filename = self.ldaMsgTable.replace('$', '.')

        word_freq = dict() #stores frequencies of each word type
        topic_freq = dict() #stores frequencies of each topic
        topic_word_freq = dict() #stores frequencies words being in a particular topics
        all_freq = 0 # stores the number of word encountered

        print("[Generating whitelist]")
        #whitelist = set(self.getWordWhiteList())
        whitelist = None
        print("[Done]")

        #update freqs based on 1 msg at a time
        print("[Finding lda messages]")
        msgs = 0
        ssMsgCursor = self.getMessages(messageTable = self.ldaMsgTable)#, where = "user_id < 2000000")
        for messageRow in ssMsgCursor: #important that this just read one line at a time
            message_id = messageRow[0]
            topicsEncoded = messageRow[1]
            if topicsEncoded:
                msgs+=1
                if msgs % 5000 == 0: #progress update
                    print(("Messages Read: %dk" % int(msgs/1000)))

                #print "encoded: %s " % str(topicsEncoded)
                wordTopics = loads(topicsEncoded)
                #print "decoded: %s" % str(wordTopics)

                for wt in wordTopics:
                   (word, topic) = (wt['term'], wt['topic_id'])
                   #update word frequencies:
                   all_freq += 1
                   if not topic in topic_freq: #this could go inside the whitelist check also, with minimal change to output
                       topic_freq[topic] = 1
                       topic_word_freq[topic] = dict()
                   else:
                       topic_freq[topic] += 1

                   #if word in whitelist:
                   if not word in word_freq:
                       word_freq[word] = 1
                   else:
                       word_freq[word] += 1
                   if not word in topic_word_freq[topic]:
                        topic_word_freq[topic][word] = 1
                   else:
                        topic_word_freq[topic][word] += 1

        #COMPUTE DISTRIBUTIONS:
        print("[Computing Distributions]")
        pTopicGivenWords = dict()
        likelihoods = dict()
        log_likelihoods = dict()
        pWordGivenTopics = dict()

        for topic in topic_freq:
            pTopic = topic_freq[topic] / float(all_freq)
            pTopicGivenWords[topic] = dict()
            likelihoods[topic] = dict()
            log_likelihoods[topic] = dict()
            pWordGivenTopics[topic] = dict()

            for word in topic_word_freq[topic]:
                pWord = word_freq[word] / float(all_freq)
                pWordTopic = topic_word_freq[topic][word] / float(all_freq)
                pTopicGivenWords[topic][word] = pWordTopic / pWord
                likelihoods[topic][word] = pWordTopic / (pWord * pTopic)
                log_likelihoods[topic][word] = log2(1+likelihoods[topic][word])
                pWordGivenTopics[topic][word] = pWordTopic / pTopic

        #print pTopicGivenWords to file
        self.printDistToCSV(pTopicGivenWords, filename+'.topicGivenWord.csv')
        #print likelihoods to file
        #self.printDistToCSV(likelihoods, filename+'.lik.csv')
        #print log_likelihoods to file
        self.printDistToCSV(log_likelihoods, filename+'.loglik.csv')
        #print word given topic
        self.printDistToCSV(pWordGivenTopics, filename+'.wordGivenTopic.csv')

        #threshold:
        newLLs = dict()
        for topic, wordDist in log_likelihoods.items():
            newLLs[topic] = dict()
            sortedWordValues = sorted(list(wordDist.items()), key = lambda w_v: w_v[1] if not isnan(w_v[1]) else -1000, reverse = True)
            threshold = 0
            if len(sortedWordValues) > 1:
                threshold = 0.50 * sortedWordValues[0][1]
            for word, value in sortedWordValues:
                if value > threshold:
                    newLLs[topic][word] = topic_word_freq[topic][word]
        self.printDistToCSV(newLLs, filename+'.freq.threshed50.loglik.csv')

        #TODO: print topics to tables:
        #id, topic, term, pcond, lik, loglik


    @staticmethod
    def printDistToCSV(dist, fileName):
        print("[Writing Distribution CSV to %s]" %fileName)
        csvWriter = csv.writer(open(fileName, 'w'))
        csvWriter.writerow(['topic_id', 'word1', 'word1_score', 'word2', 'word2_score', '...'])
        for topic in sorted(list(dist.keys()), key = lambda k: int(k) if str(k).isdigit() else k):
            wordDist = dist[topic]
            row = [topic]
            for wordValue in sorted(list(wordDist.items()), key = lambda w_v1: w_v1[1] if not isnan(w_v1[1]) else -1000, reverse = True):
                row.extend(wordValue)
            csvWriter.writerow(row)


    def getWordWhiteList(self, pocc = 0.01):
        wg = self.getWordGetterPOcc(pocc)
        return wg.getDistinctFeatures()


    def addLDAFeatTable(self, ldaMessageTable, tableName = None, valueFunc = lambda d: d):
        """Creates feature tuples (correl_field, feature, values) table where features are ngrams"""
        """Optional ValueFunc program scales that features by the function given"""
        #CREATE TABLE:
        featureName =  'lda'+'_'+ldaMessageTable.split('$')[1]
        featureTableName = self.createFeatureTable(featureName, 'SMALLINT UNSIGNED', 'INTEGER', tableName, valueFunc)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        usql = """SELECT %s FROM %s GROUP BY %s""" % (
            self.correl_field, self.corptable, self.correl_field)
        msgs = 0#keeps track of the number of messages read
        cfRows = self._executeGetList(usql)#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        self._disableTableKeys(featureTableName)#for faster, when enough space for repair by sorting
        for cfRow in cfRows:
            cf_id = cfRow[0]
            mids = set() #currently seen message ids
            freqs = dict() #holds frequency of n-grams
            totalInsts = 0 #total number of (non-distinct) topics

            #grab topics by messages for that cf:
            for messageRow in self.getMessagesForCorrelField(cf_id, messageTable = ldaMessageTable):
                message_id = messageRow[0]
                topicsEncoded = messageRow[1]
                if not message_id in mids and topicsEncoded:
                    msgs+=1
                    if msgs % PROGRESS_AFTER_ROWS == 0: #progress update
                        dlac.warn("Messages Read: %dk" % int(msgs/1000))
                    #print topicsEncoded
                    topics = json.loads(topicsEncoded)

                    for topic in topics:
                        totalInsts += 1
                        topicId = topic['topic_id']
                        if not topicId in freqs:
                            freqs[topicId] = 1
                        else:
                            freqs[topicId] += 1
                    mids.add(message_id)

            #write topic to database (no need for "REPLACE" because we are creating the table)
            wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
            totalInsts = float(totalInsts) #to avoid casting each time below
            rows = [(k, v, valueFunc((v / totalInsts))) for k, v in freqs.items() ] #adds group_norm and applies freq filter
            self._executeWriteMany(wsql, rows)

        dlac.warn("Done Reading / Inserting.")

        dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
        self._enableTableKeys(featureTableName)#rebuilds keys
        dlac.warn("Done\n")
        return featureTableName;


class LDAEstimator(object):
    def __init__(self, feature_getter, num_topics, alpha, beta, iterations, num_stopwords=50, no_stopping=False):
        self.feature_getter = feature_getter
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.num_stopwords = num_stopwords
        self.no_stopping = no_stopping

        self._stopwords = None

    @property
    def stopwords(self):
        if self._stopwords is None:
            self._stopwords = set()
            if not self.no_stopping:
                top_feats = self.feature_getter.getTopFeats(n=self.num_stopwords)
                for top_feat_row in top_feats:
                    self._stopwords.add(top_feat_row[0])
                print('Automatically removed stopwords: {}'.format(str(self._stopwords)))
        return self._stopwords

    def estimate_topics(self, feature_lines_file, mallet_path=None):
        if not mallet_path:
            print('Estimating LDA topics using PyMallet.')
            estimate_topics(feature_lines_file, num_topics=self.num_topics, alpha=self.alpha, beta=self.beta,
                            iterations=self.iterations, stoplist=self.stopwords)
            state_file = defaults.OUTPUT_STATE_FILE
        else:
            print('Estimating LDA topics using Mallet.')
            id2word = corpora.Dictionary(line for line in self._load_corpus(feature_lines_file))
            mallet = LdaMallet(mallet_path, corpus=self._load_corpus(feature_lines_file, dictionary=id2word),
                               id2word=id2word, num_topics=self.num_topics, alpha=self.alpha,
                               iterations=self.iterations)
            state_file = mallet.fstate()
        return state_file

    def _load_corpus(self, feature_lines_file, dictionary=None):
        token_regex = r'(#|@)?(?!(\W)\2+)([a-zA-Z\_\-\'0-9\(-\@]{2,})'
        with open(feature_lines_file) as f:
            for line in f:
                _, _, line = line.split(' ', 2)
                tokens = [token.group(0) for token in re.finditer(token_regex, line) if token.group(0) not in self.stopwords]
                if dictionary is None:
                    yield tokens
                else:
                    yield dictionary.doc2bow(tokens)
