#!/usr/bin/env python
#########################################
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import re
from numpy import log2, isnan
import csv
import os.path, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)).replace("/dlatk/LexicaInterface",""))

from dlatk.featureExtractor import FeatureExtractor
from dlatk import featureWorker
from dlatk import fwConstants as fwc

from json import loads


DEF_LDA_MSG_TABLE = 'messages_en_lda$msgs_en_tok_a30'

class LDAExtractor(FeatureExtractor):

    def __init__(self, corpdb=fwc.DEF_CORPDB, corptable=fwc.DEF_CORPTABLE, correl_field=fwc.DEF_CORREL_FIELD, 
                 mysql_host = "localhost", message_field=fwc.DEF_MESSAGE_FIELD, messageid_field=fwc.DEF_MESSAGEID_FIELD, 
                 encoding=fwc.DEF_ENCODING, use_unicode=fwc.DEF_UNICODE_SWITCH, ldaMsgTable = DEF_LDA_MSG_TABLE):
        super(LDAExtractor, self).__init__(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode)
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
        self.printDistToCSV(likelihoods, filename+'.lik.csv')
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
        csvWriter = csv.writer(open(fileName, 'wb'))
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
        _warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
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
                        _warn("Messages Read: %dk" % int(msgs/1000))
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
        
        _warn("Done Reading / Inserting.")

        _warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
        self._enableTableKeys(featureTableName)#rebuilds keys
        _warn("Done\n")
        return featureTableName;



##################################################################
## Main methods


class LDAExtractorParser(ArgumentParser):

    def __init__(self, description="On Features Class.", prefix_chars='-+', formatter_class=ArgumentDefaultsHelpFormatter):
        ## Argument Parser ##

        super(LDAExtractorParser,self).__init__(description=description, prefix_chars=prefix_chars, formatter_class=formatter_class)

        # group = self.add_argument_group('Save / Load Program States', '')
        # group.add_argument('-l', '--load', metavar='NAME', dest='load', type=str,
        #                    help='Load the state of any objects in the given file (done first).')
        # group.add_argument('-s', '--save', metavar='NAME', dest='save', type=str,
        #                    help='Saves the state of any open objects (done last).')
        # group.add_argument('-ls', '--list_states', action='store_true', dest='liststates',
        #                    help='List all of the possible state names.')
        # group.add_argument('--savedir', metavar='DIR', dest='savedir', type=str, default=_DEF_SAVEDIR,
        #                    help='Directory from which to load and save states.')
        # group.add_argument('--drop', metavar='objName', dest='drop', type=str, action='append', default=[],
        #                    help='Specify the abbreviation of an object to drop (ol, lf, lc) so it is recreated from scratch')
        # group.add_argument('--print', action='store_true', dest='printObj',
        #                    help='Print any loaded objects')

        group = self.add_argument_group('Corpus Variables', 'Defining the data from which features are extracted.')
        group.add_argument('-H', '--host', metavar='HOST', dest='host', default=fwc.MYSQL_HOST,
                           help='Host that contains the mysql dbs and tables')
        group.add_argument('-d', '--corpdb', metavar='DB', dest='corpdb', default=fwc.DEF_CORPDB,
                            help='Corpus Database Name.')
        group.add_argument('-t', '--corptable', metavar='TABLE', dest='corptable', default=fwc.DEF_CORPTABLE,
                            help='Corpus Table.')
        group.add_argument('-c', '--correl_field', metavar='FIELD', dest='correl_field', default=fwc.DEF_CORREL_FIELD,
                            help='Correlation Field (AKA Group Field): The field which features are aggregated over.')
        group.add_argument('--message_field', metavar='FIELD', dest='message_field', default=fwc.DEF_MESSAGE_FIELD,
                            help='The field where the text to be analyzed is located.')
        group.add_argument('--messageid_field', metavar='FIELD', dest='messageid_field', default=fwc.DEF_MESSAGEID_FIELD,
                            help='The unique identifier for the message.')


        group = self.add_argument_group('LDA Extraction Variables', '')
        group.add_argument('-m', '--lda_msg_tbl', metavar='TABLE', dest='ldamsgtbl', type=str, default=DEF_LDA_MSG_TABLE,
                           help='LDA Message Table')


        group = self.add_argument_group('LDA Extraction Actions', '')
        group.add_argument('--create_dists', action='store_true', dest='createdists', 
                           help='Create conditional prob, and likelihood distributions.')


        ## Initialize any Objects ##
        self.ldae = None 




    def processLDAExtractor(self, args):
        """Main argument processing area"""
        ##Add Argument Processing here
        
        self.ldae = LDAExtractor(args.corpdb, args.corptable, args.correl_field, args.host, args.message_field, args.messageid_field, fwc.DEF_ENCODING, fwc.DEF_UNICODE_SWITCH, args.ldamsgtbl)

        if args.createdists:
            self.ldae.createDistributions()


    def getParser(self):
        """Just incase someone is confused by the inheritance"""
        return self

    ###### Shouldn't Need to Edit Below Here ########

    ### These methods can be overWritten when creating subclass ###

    def processLoad(self, args):
        """processing state load arguments"""
        #seperated to make things easier for other to inherit this module
        #(can be overridden)

        #load:
        if args.load:
            objTup = self.load(args.savedir, args.load)
            if isinstance(objTup, tuple):
                (self.ol, self.lf) = objTup
            else:
                print("Nothing to load")     

        if args.liststates:
            self.printstates(args.savedir)

        if args.drop:
            self.drop(args.drop)

    def processSave(self, args):
        """processes the save arguments"""
        #separated to make it easy for others to put last
        #(can be overridden)
        if args.printObj:
            self.printObjs()

        if args.save:
            self.save(args.savedir, args.save, (self.ol, self.lf))

    def processArgs(self, args = ''):
        """Processes all arguments"""

        ##LOAD ARGUMENTS##
        if not args: 
            args = self.parse_args()

        ##PROCESS ARGUMENTS##
        # self.processLoad(args)
        self.processLDAExtractor(args)
        # self.processSave(args)

    def printObjs(self):
        if self.ol and not self.lf:
            print("OntoNotes Data:")
            print(self.ol)
        if self.lf:
            print("Language Features:")
            print(self.lf)

    def drop(self, objNames):
        for name in objNames:
            self.__dict__[name] = None

            
    ##HELPER FUNCTIONS:
    saveExtension = 'pickle'
    @staticmethod
    def save(savedir, savename, objectTup):
        saveFile = "%s/%s.%s" % (savedir, savename, OnFeaturesParser.saveExtension)
        print("Saving state to: %s" % saveFile)
        pickle.dump( objectTup, open( saveFile, "wb" ) )

    @staticmethod
    def load(savedir, savename):
        saveFile = "%s/%s.%s" % (savedir, savename, OnFeaturesParser.saveExtension)
        print("Loading state from: %s" % saveFile)
        objectTup = pickle.load( open( saveFile, "rb" ) )
        return objectTup

    @staticmethod
    def removeStateFile(savedir, savename):
        saveFile = "%s/%s.%s" % (savedir, savename, OnFeaturesParser.saveExtension)
        print("Removing state file: %s" % saveFile)
        os.remove(saveFile)

    matchExtension = re.compile(r'^(.*)\.'+saveExtension+'$')
    @staticmethod
    def printstates(savedir):
        files = os.listdir(savedir)
        names = []
        for fname in files:
            mObj = OnFeaturesParser.matchExtension.match(fname)
            if mObj:
                if os.stat("%s/%s"%(savedir,fname)).st_size > 0:
                    names.append((mObj.group(1), int(os.stat("%s/%s"%(savedir,fname)).st_size /1048576) ))
        if names:
            print("\nThe following saved states are available for loading:\n")
            print("  %-36s %12s" % ('NAME', 'SIZE'))
            print("  %-36s %12s" % ('----', '----'))
            for tup in sorted(names, key=lambda t: t[0]):
                print("  %-36s %10dMB" % tup)
        else:
            print("\nNo saved states available in directory: %s" % savedir)


###########################################################################
###########################################################################
if __name__ == "__main__":
    parser = LDAExtractorParser()
    parser.processArgs()

