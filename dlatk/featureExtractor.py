import re
import json
import sys
import csv
import gzip
import datetime
from pprint import pprint
from dateutil.parser import parse as dtParse
from collections import Counter
import traceback
from xml.dom.minidom import parseString as xmlParseString
from datetime import timedelta


#math / stats:
from math import floor, log10
from numpy import mean, std
#import imp

#nltk
try:
    from nltk.tree import ParentedTree
    from nltk.corpus import wordnet as wn
except ImportError:
    print("warning: unable to import nltk.tree or nltk.corpus or nltk.data")

#infrastructure
from .dlaWorker import DLAWorker
from . import dlaConstants as dlac
from . import textCleaner as tc
from .mysqlMethods import mysqlMethods as mm

#local / nlp
from .lib.happierfuntokenizing import Tokenizer #Potts tokenizer

try:
    import jsonrpclib
    from simplejson import loads
except ImportError:
    dlac.warn("Cannot import jsonrpclib or simplejson (cannot use addPOSAndTimexDiffFeatTable)")
    pass
try:
    from textstat.textstat import textstat
except ImportError:
    dlac.warn("Cannot import textstat (cannot use addFleschKincaidTable)")
    pass

#feature extractor constants:
offsetre = re.compile(r'p(\-?\d+)([a-z])')
toffsetre = re.compile(r'pt(\-?\d+)([a-z])')
TimexDateTimeTypes = frozenset(['date', 'time'])

class FeatureExtractor(DLAWorker):
    """Deals with extracting features from text and writing tables of features

    Returns
    -------
    FeatureExtractor object

    Examples
    --------
    Extract 1, 2 and 3 grams

    >>> for n in xrange(1, 4):
    ...     fe.addNGramTable(n=n)
    """

    ##INSTANCE METHODS##

    def addTopicLexFromTopicFile(self, topicfile, newtablename, topiclexmethod, threshold):
        """Creates a lexicon from a topic file

        Parameters
        ----------
        topicfile : str
            Name of topic file to use to build the topic lexicon.
        newtablename : str
            New (topic) lexicon name.
        topiclexmethod : str
            must be one of: "csv_lik", "standard".
        threshold : float
            Default = float('-inf').

        Returns
        -------
        newtablename : str
            New (topic) lexicon name.
        """
        topiclex = None
        if topiclexmethod=='standard':
            topiclex = interface.Lexicon(interface.loadLexiconFromTopicFile(topicfile))
            topiclex.createLexiconTable(newtablename)
        elif topiclexmethod=='csv_lik':
            topiclex = interface.WeightedLexicon(interface.loadWeightedLexiconFromTopicCSV(topicfile, threshold))
            topiclex.createWeightedLexiconTable(newtablename)
        return newtablename



    ##Feature Tables ##

    def addNGramTable(self, n, lowercase_only=dlac.LOWERCASE_ONLY, min_freq=1, tableName = None, valueFunc = lambda d: d, metaFeatures = True, extension = None):
        """Creates feature tuples (correl_field, feature, values) table where features are ngrams

        Parameters
        ----------
        n : int
            ?????
        lowercase_only : boolean
            use only lowercase charngrams if True
        min_freq : :obj:`int`, optional
            ?????
        tableName : :obj:`str`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given
        metaFeatures : :obj:`boolean`, optional
            ?????

        Returns
        -------
        featureTableName : str
            Name of n-gram table: feat%ngram%corptable%correl_field%transform
        """
        ##NOTE: correl_field should have an index for this to be quick
        tokenizer = Tokenizer(use_unicode=self.use_unicode)

        #debug:
        #print "valueFunc(30) = %f" % valueFunc(float(30)) #debug

        #CREATE TABLE:
        featureName = str(n)+'gram'
        varcharLength = min((dlac.VARCHAR_WORD_LENGTH-(n-1))*n, 255)
        featureTableName = self.createFeatureTable(featureName, "VARCHAR(%d)"%varcharLength, 'INTEGER', tableName, valueFunc, extension = extension)

        if metaFeatures:
            # If metafeats is on, make a metafeature table as well
            mfLength = 16
            mfName = "meta_"+featureName
            mfTableName = self.createFeatureTable(mfName, "VARCHAR(%d)" % mfLength, 'INTEGER', tableName, valueFunc, extension = extension)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        usql = """SELECT %s FROM %s GROUP BY %s""" % (
            self.correl_field, self.corptable, self.correl_field)
        msgs = 0 # keeps track of the number of messages read
        cfRows = FeatureExtractor.noneToNull(mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode))#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows)*n < dlac.MAX_TO_DISABLE_KEYS: mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting

        #warnedMaybeForeignLanguage = False
        for cfRow in cfRows:
            cf_id = cfRow[0]

            mids = set() #currently seen message ids
            freqs = dict() #holds frequency of n-grams
            totalGrams = 0 #total number of (non-distinct) n-grams seen for this user
            totalChars = 0

            #grab n-grams by messages for that cf:

            for messageRow in self.getMessagesForCorrelField(cf_id, warnMsg = False):
                message_id = messageRow[0]
                message = messageRow[1]
                if not message_id in mids and message:
                    msgs+=1
                    if msgs % dlac.PROGRESS_AFTER_ROWS == 0: #progress update
                        dlac.warn("Messages Read: %dk" % int(msgs/1000))
                    message = tc.treatNewlines(message)
                    message = tc.shrinkSpace(message)

                    #words = message.split()
                    if not self.use_unicode:
                        words = [tc.removeNonAscii(w) for w in tokenizer.tokenize(message)]
                    else:
                        words = [tc.removeNonUTF8(w) for w in tokenizer.tokenize(message)]

                    gram = '' ## MAARTEN
                    for i in range(0,(len(words) - n)+1):
                        totalGrams += 1
                        gram = ' '.join(words[i:i+n])
                        #truncate:
                        gram = gram[:varcharLength]

                        if lowercase_only: gram = gram.lower()

                        try:
                            freqs[gram] += 1
                        except KeyError:
                            freqs[gram] = 1

                        if metaFeatures:
                            totalChars += len(gram)
                    mids.add(message_id)
                    if metaFeatures:
                        # why is this in here?
                        totalChars += len(gram)

            #write n-grams to database (no need for "REPLACE" because we are creating the table)
            if totalGrams:
                insert_idx_start = 0
                insert_idx_end = dlac.MYSQL_BATCH_INSERT_SIZE
                wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
                totalGrams = float(totalGrams) # to avoid casting each time below
                try:
                    if self.use_unicode:
                        rows = [(k, v, valueFunc((v / totalGrams))) for k, v in freqs.items() if v >= min_freq] #adds group_norm and applies freq filter
                    else:
                        rows = [(k.encode('utf-8'), v, valueFunc((v / totalGrams))) for k, v in freqs.items() if v >= min_freq] #adds group_norm and applies freq filter
                except:
                    print([k for k, v in freqs.items()])
                    sys.exit()
                while insert_idx_start < len(rows):
                    insert_rows = rows[insert_idx_start:min(insert_idx_end, len(rows))]
                    #_warn("Inserting rows %d to %d... " % (insert_idx_start, insert_idx_end))
                    mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, insert_rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode);
                    insert_idx_start += dlac.MYSQL_BATCH_INSERT_SIZE
                    insert_idx_end += dlac.MYSQL_BATCH_INSERT_SIZE



                wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
                totalGrams = float(totalGrams) # to avoid casting each time below
                if self.use_unicode:
                    rows = [(k, v, valueFunc((v / totalGrams))) for k, v in freqs.items() if v >= min_freq] #adds group_norm and applies freq filter
                else:
                    rows = [(k.encode('utf-8'), v, valueFunc((v / totalGrams))) for k, v in freqs.items() if v >= min_freq] #adds group_norm and applies freq filter

                if metaFeatures:
                    mfRows = []
                    mfwsql = """INSERT INTO """+mfTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
                    avgGramLength = totalChars / totalGrams
                    avgGramsPerMsg = totalGrams / len(mids)
                    mfRows.append( ('_avg'+str(n)+'gramLength', avgGramLength, valueFunc(avgGramLength)) )
                    mfRows.append( ('_avg'+str(n)+'gramsPerMsg', avgGramsPerMsg, valueFunc(avgGramsPerMsg)) )
                    mfRows.append( ('_total'+str(n)+'grams', totalGrams, valueFunc(totalGrams)) )
                    mm.executeWriteMany(self.corpdb, self.dbCursor, mfwsql, mfRows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

                # mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding)

        dlac.warn("Done Reading / Inserting.")

        if len(cfRows)*n < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys
        dlac.warn("Done\n")
        return featureTableName


    def addCharNGramTable(self, n, lowercase_only=dlac.LOWERCASE_ONLY, min_freq=1, tableName = None, valueFunc = lambda d: d, metaFeatures = True):
        """Extract character ngrams from a message table

        Parameters
        ----------
        n : int
            ?????
        lowercase_only : boolean
            use only lowercase charngrams if True
        min_freq : :obj:`int`, optional
            ?????
        tableName : :obj:`str`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given
        metaFeatures : :obj:`boolean`, optional
            ?????

        Returns
        -------
        featureTableName : str
            Name of n-gram table: feat%nCgram%corptable%correl_field%transform
        """
        ##NOTE: correl_field should have an index for this to be quick
        #tokenizer = Tokenizer(use_unicode=self.use_unicode)

        #debug:
        #print "valueFunc(30) = %f" % valueFunc(float(30)) #debug

        #CREATE TABLE:
        featureName = str(n)+'Cgram'
        varcharLength = min((dlac.VARCHAR_WORD_LENGTH-(n-1))*n, 255)
        featureTableName = self.createFeatureTable(featureName, "VARCHAR(%d)"%varcharLength, 'INTEGER', tableName, valueFunc)

        if metaFeatures:
            # If metafeats is on, make a metafeature table as well
            mfLength = 16
            mfName = "meta_"+featureName
            mfTableName = self.createFeatureTable(mfName, "VARCHAR(%d)" % mfLength, 'INTEGER', tableName, valueFunc)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        usql = """SELECT %s FROM %s GROUP BY %s""" % (
            self.correl_field, self.corptable, self.correl_field)
        msgs = 0 # keeps track of the number of messages read
        cfRows = FeatureExtractor.noneToNull(mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode))#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows)*n < dlac.MAX_TO_DISABLE_KEYS: mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting

        #warnedMaybeForeignLanguage = False
        for cfRow in cfRows:
            cf_id = cfRow[0]

            mids = set() #currently seen message ids
            freqs = dict() #holds frequency of n-grams
            totalGrams = 0 #total number of (non-distinct) n-grams seen for this user
            totalChars = 0

            #grab n-grams by messages for that cf:

            for messageRow in self.getMessagesForCorrelField(cf_id, warnMsg = False):
                message_id = messageRow[0]
                message = messageRow[1]
                if not message_id in mids and message:
                    msgs+=1
                    if msgs % dlac.PROGRESS_AFTER_ROWS == 0: #progress update
                        dlac.warn("Messages Read: %dk" % int(msgs/1000))
                    message = tc.treatNewlines(message)
                    message = tc.shrinkSpace(message)

                    #words = message.split()
                    if self.use_unicode:
                        words = [tc.removeNonUTF8(w) for w in list(message)]
                    else:
                        words = [tc.removeNonAscii(w) for w in list(message)]

                    gram = '' ## MAARTEN
                    for i in range(0,(len(words) - n)+1):
                        totalGrams += 1
                        gram = ' '.join(words[i:i+n])
                        #truncate:
                        gram = gram[:varcharLength]
                        if lowercase_only: gram = gram.lower()

                        try:
                            freqs[gram] += 1
                        except KeyError:
                            freqs[gram] = 1

                        if metaFeatures:
                            totalChars += len(gram)
                    mids.add(message_id)
                    if metaFeatures:
                        # why is this in here?
                        totalChars += len(gram)

            #write n-grams to database (no need for "REPLACE" because we are creating the table)
            if totalGrams:
                insert_idx_start = 0
                insert_idx_end = dlac.MYSQL_BATCH_INSERT_SIZE
                wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
                totalGrams = float(totalGrams) # to avoid casting each time below
                if self.use_unicode:
                    rows = [(k, v, valueFunc((v / totalGrams))) for k, v in freqs.items() if v >= min_freq] #adds group_norm and applies freq filter
                else:
                    rows = [(k.encode('utf-8'), v, valueFunc((v / totalGrams))) for k, v in freqs.items() if v >= min_freq] #adds group_norm and applies freq filter

                while insert_idx_start < len(rows):
                    insert_rows = rows[insert_idx_start:min(insert_idx_end, len(rows))]
                    #_warn("Inserting rows %d to %d... " % (insert_idx_start, insert_idx_end))
                    mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, insert_rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode);
                    insert_idx_start += dlac.MYSQL_BATCH_INSERT_SIZE
                    insert_idx_end += dlac.MYSQL_BATCH_INSERT_SIZE



                wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
                totalGrams = float(totalGrams) # to avoid casting each time below
                if self.use_unicode:
                    rows = [(k, v, valueFunc((v / totalGrams))) for k, v in freqs.items() if v >= min_freq] #adds group_norm and applies freq filter
                else:
                    rows = [(k.encode('utf-8'), v, valueFunc((v / totalGrams))) for k, v in freqs.items() if v >= min_freq] #adds group_norm and applies freq filter
                if metaFeatures:
                    mfRows = []
                    mfwsql = """INSERT INTO """+mfTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
                    avgGramLength = totalChars / totalGrams
                    avgGramsPerMsg = totalGrams / len(mids)
                    mfRows.append( ('_avg'+str(n)+'gramLength', avgGramLength, valueFunc(avgGramLength)) )
                    mfRows.append( ('_avg'+str(n)+'gramsPerMsg', avgGramsPerMsg, valueFunc(avgGramsPerMsg)) )
                    mfRows.append( ('_total'+str(n)+'grams', totalGrams, valueFunc(totalGrams)) )
                    mm.executeWriteMany(self.corpdb, self.dbCursor, mfwsql, mfRows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

                # mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding)

        dlac.warn("Done Reading / Inserting.")

        if len(cfRows)*n < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys
        dlac.warn("Done\n")
        return featureTableName


    def addNGramTableFromTok(self, n, lowercase_only=dlac.LOWERCASE_ONLY, min_freq=1, tableName = None, valueFunc = lambda d: d, metaFeatures = True):
        """???

        Parameters
        ----------
        n : int
            ?????
        lowercase_only : boolean
            use only lowercase charngrams if True
        min_freq : :obj:`int`, optional
            ?????
        tableName : :obj:`str`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given
        metaFeatures : :obj:`boolean`, optional
            ?????

        Returns
        -------
        featureTableName : str
            Name of n-gram table: feat%nCgram%corptable%correl_field%transform
        """
        ##NOTE: correl_field should have an index for this to be quick

        #debug:
        #print "valueFunc(30) = %f" % valueFunc(float(30)) #debug


        #CREATE TABLE:
        featureName = str(n)+'gram'
        varcharLength = min((dlac.VARCHAR_WORD_LENGTH-(n-1))*n, 255)
        featureTableName = self.createFeatureTable(featureName, "VARCHAR(%d)"%varcharLength, 'INTEGER', tableName, valueFunc)

        if metaFeatures:
            # If metafeats is on, make a metafeature table as well
            mfLength = 16
            mfName = "meta_"+featureName
            mfTableName = self.createFeatureTable(mfName, "VARCHAR(%d)" % mfLength, 'INTEGER', tableName, valueFunc)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        usql = """SELECT %s FROM %s GROUP BY %s""" % (
            self.correl_field, self.corptable, self.correl_field)
        msgs = 0 # keeps track of the number of messages read
        cfRows = FeatureExtractor.noneToNull(mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode))#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows)*n < dlac.MAX_TO_DISABLE_KEYS: mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting

        for cfRow in cfRows:
            cf_id = cfRow[0]

            mids = set() #currently seen message ids
            freqs = dict() #holds frequency of n-grams
            totalGrams = 0 #total number of (non-distinct) n-grams seen for this user
            totalChars = 0

            #grab n-grams by messages for that cf:
            for messageRow in self.getMessagesForCorrelField(cf_id, warnMsg = False):
                message_id = messageRow[0]
                json_tokens = messageRow[1]
                if not message_id in mids and json_tokens:
                    msgs+=1
                    if msgs % dlac.PROGRESS_AFTER_ROWS == 0: #progress update
                        dlac.warn("Messages Read: %dk" % int(msgs/1000))

                    words = None
                    try:
                        words = json.loads(json_tokens)
                    except ValueError as e:
                        raise ValueError(str(e)+"Your message table is either not tokenized (use --add_ngrams) or there might be something else wrong.")

                    gram = '' ## MAARTEN
                    for i in range(0,(len(words) - n)+1):
                        totalGrams += 1
                        gram = ' '.join(words[i:i+n])
                        #truncate:
                        gram = gram[:varcharLength]
                        if lowercase_only: gram = gram.lower()

                        try:
                            freqs[gram] += 1
                        except KeyError:
                            freqs[gram] = 1

                        if metaFeatures:
                            totalChars += len(gram)
                    mids.add(message_id)
                    if metaFeatures:
                        # why is this in here?
                        totalChars += len(gram)

            #write n-grams to database (no need for "REPLACE" because we are creating the table)
            #pprint(freqs)
            if totalGrams:
                wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
                totalGrams = float(totalGrams) # to avoid casting each time below
                if self.use_unicode:
                    rows = [(k, v, valueFunc((v / totalGrams))) for k, v in freqs.items() if v >= min_freq] #adds group_norm and applies freq filter
                else:
                    rows = [(k.encode('utf-8'), v, valueFunc((v / totalGrams))) for k, v in freqs.items() if v >= min_freq] #adds group_norm and applies freq filter
                if metaFeatures:
                    mfRows = []
                    mfwsql = """INSERT INTO """+mfTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
                    avgGramLength = totalChars / totalGrams
                    avgGramsPerMsg = totalGrams / len(mids)
                    mfRows.append( ('_avg'+str(n)+'gramLength', avgGramLength, valueFunc(avgGramLength)) )
                    mfRows.append( ('_avg'+str(n)+'gramsPerMsg', avgGramsPerMsg, valueFunc(avgGramsPerMsg)) )
                    mfRows.append( ('_total'+str(n)+'grams', totalGrams, valueFunc(totalGrams)) )
                    mm.executeWriteMany(self.corpdb, self.dbCursor, mfwsql, mfRows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                #print "\n\n\nROWS TO ADD!!"
                #pprint(rows) #DEBUG
                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding)

        dlac.warn("Done Reading / Inserting.")

        if len(cfRows)*n < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys
        dlac.warn("Done\n")
        return featureTableName


    def _getCollocsFromTable(self, colloc_table, pmi_filter_thresh, colloc_column, pmi_filter_column):
        res = colloc_table.split('.')
        if len(res) == 1:
            colloc_schema = self.lexicondb
        if len(res) == 2:
            (colloc_schema, colloc_table) = res
        elif len(res) > 2:
            raise "Invalid collocation table name."

        has_column_query = "SELECT * FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME = '{}' AND COLUMN_NAME = '{}'".format(colloc_schema, colloc_table, pmi_filter_column)
        res = mm.executeGetList(self.corpdb, self.dbCursor, has_column_query, charset=self.encoding, use_unicode=self.use_unicode)
        has_pmi_column = len(res) > 0
        if has_pmi_column:
            res = mm.executeGetList(self.corpdb, self.dbCursor, "SELECT {} FROM {}.{} WHERE {} < {}".format(colloc_column, colloc_schema, colloc_table, pmi_filter_thresh, pmi_filter_column), charset=self.encoding, use_unicode=self.use_unicode)
        else:
            dlac.warn("No column named {} found.  Using all collocation in table.".format(pmi_filter_column))
            res = mm.executeGetList(self.corpdb, self.dbCursor, "SELECT {} FROM {}.{}".format(colloc_column, colloc_schema, colloc_table), charset=self.encoding, use_unicode=self.use_unicode)
        return [row[0] for row in res]

    def _countFeatures(self, collocSet, maxCollocSizeByFirstWord, message, tokenizer, freqs, lowercase_only=dlac.LOWERCASE_ONLY, includeSubCollocs=False):
        '''?????

        Parameters
        ----------
        collocSet : set
            set of features we will extract
        maxCollocSizeByFirstWord : dict
            ?????
        message : str
            text from which we would like to extract features
        tokenizer : object
            an object with method tokenize() that returns words from a piece of text
        freqs : dict
            ?????
        lowercase_only : boolean
            use only lowercase charngrams if True
            running count of how many times each word is used
        includeSubCollocs : :obj:`boolean`, optional
            ?????

        '''
        ###### BEGIN extract to new function
        message = tc.treatNewlines(message)
        if self.use_unicode:
            message = tc.removeNonUTF8(message)
        else:
            message = tc.removeNonAscii(message) #TODO: don't use for foreign languages
        message = tc.shrinkSpace(message)

        #TODO - update this to a word based dict, eg maxCollocSize[word[i]]
        maxCollocSize = 5
        #TODO - update this from the outside!!!
        varcharLength = 128

        words = tokenizer.tokenize(message)
        gram = ''
        (window_start, window_end) = (0, 0)

        while window_start < len(words):
            firstWord = words[window_start]
            if firstWord in maxCollocSizeByFirstWord:
                maxCollocSize = maxCollocSizeByFirstWord[firstWord]
            else:
                maxCollocSize = 1
            window_end = min(len(words), window_start + maxCollocSize - 1)
            while window_start <= window_end:
                potentialColloc = ' '.join(words[window_start:window_end+1])
                is1gram = window_start == window_end
                if is1gram or (potentialColloc in collocSet):
                    gram = potentialColloc
                    gram = gram[:varcharLength]
                    if lowercase_only: gram = gram.lower()
                    if gram in freqs:
                        freqs[gram] += 1
                    else:
                        freqs[gram] = 1
                    if (not is1gram) and includeSubCollocs:
                        window_end -= 1
                    else:
                        window_start = window_end + 1
                else:
                    window_end -= 1


    def addCollocFeatTable(self, collocList, lowercase_only=dlac.LOWERCASE_ONLY, min_freq=1, tableName = None, valueFunc = lambda d: d, includeSubCollocs=False, featureTypeName=None):
        """???

        Parameters
        ----------
        collocList : list
            ?????
        min_freq : :obj:`int`, optional
            ?????
        tableName : :obj:`str`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given
        includeSubCollocs : :obj:`boolean`, optional
            ?????
        featureTypeName : :obj:`?????`, optional
            ?????

        Returns
        -------
        featureTableName : str
            Name of n-gram table: ?????
        """
        ##NOTE: correl_field should have an index for this to be quick
        tokenizer = Tokenizer(use_unicode=self.use_unicode)

        #gather some summary data about collocList
        maxCollocSize = 0
        maxCollocSizeByFirstWord = {}

        for colloc in collocList:
            collocWords = colloc.split(' ')
            collocSize = len(collocWords)
            firstWord = collocWords[0]

            #update maxColloSize
            if collocSize > maxCollocSize:
                maxCollocSize = collocSize

            #update maxCollocSizeByFirstWord
            if (not (firstWord in maxCollocSizeByFirstWord)) or maxCollocSizeByFirstWord[firstWord] < collocSize:
                maxCollocSizeByFirstWord[firstWord] = collocSize

        collocSet = frozenset(collocList)

        #CREATE TABLE:
        varcharLength = min(dlac.VARCHAR_WORD_LENGTH*5, 255)
        featureTableName = self.createFeatureTable(featureTypeName, "VARCHAR(%d)"%varcharLength, 'INTEGER', tableName, valueFunc)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        usql = """SELECT %s FROM %s GROUP BY %s""" % (
            self.correl_field, self.corptable, self.correl_field)
        msgs = 0 # keeps track of the number of messages read
        cfRows = FeatureExtractor.noneToNull(mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode))#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS: mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting

        for cfRow in cfRows:
            cf_id = cfRow[0]

            mids = set() #currently seen message ids
            freqs = dict() #holds frequency of n-grams
            totalGrams = 0 #total number of (non-distinct) n-grams seen for this user

            #grab n-grams by messages for that cf:
            for messageRow in self.getMessagesForCorrelField(cf_id, warnMsg = False):
                message_id = messageRow[0]
                message = messageRow[1]
                if not message_id in mids and message:
                    msgs+=1
                    if msgs % dlac.PROGRESS_AFTER_ROWS == 0: #progress update
                        dlac.warn("Messages Read: %dk" % int(msgs/1000))

                    #TODO: remove if keeping other characters
                    message = tc.treatNewlines(message)
                    if self.use_unicode:
                        message = tc.removeNonUTF8(message) #TODO: don't use for foreign languages
                    else:
                        message = tc.removeNonAscii(message) #TODO: don't use for foreign languages
                    message = tc.shrinkSpace(message)

                    self._countFeatures(collocSet, maxCollocSizeByFirstWord, message, tokenizer, freqs, lowercase_only, includeSubCollocs)
                    #TODO - save this somewhere?  Accumulate for all  messages... way to sum hash tables?  Or just pass it in?

                    mids.add(message_id)
                    ##Selah: here is where it ends

            totalGrams = sum(freqs.values())
            #write n-grams to database (no need for "REPLACE" because we are creating the table)
            if totalGrams:
                wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
                totalGrams = float(totalGrams) # to avoid casting each time below
                if self.use_unicode:
                    rows = [(k, v, valueFunc((v / totalGrams))) for k, v in freqs.items() if v >= min_freq] #adds group_norm and applies freq filter
                else:
                    rows = [(k.encode('utf-8'), v, valueFunc((v / totalGrams))) for k, v in freqs.items() if v >= min_freq] #adds group_norm and applies freq filter

                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

        dlac.warn("Done Reading / Inserting.")

        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys
        dlac.warn("Done\n")
        return featureTableName

    def addNGramTableGzipCsv(self, n, gzCsv, idxMsgField, idxIdField, idxCorrelField, lowercase_only=dlac.LOWERCASE_ONLY, min_freq=1, tableName = None, valueFunc = lambda d: d):
        """???
        This assumes each row is a unique message, originally meant for twitter

        Parameters
        ----------
        n : int
            ?????
        gzCsv : ?????
            ?????
        idxMsgField : ?????
            ?????
        idxIdField : ?????
            ?????
        idxCorrelField : ?????
            ?????
        lowercase_only : boolean
            use only lowercase charngrams if True
        min_freq : :obj:`int`, optional
            ?????
        tableName : :obj:`str`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given

        Returns
        -------
        featureTableName : str
            Name of n-gram table: ?????
        """

        tokenizer = Tokenizer(use_unicode=self.use_unicode)

        #CREATE TABLE:
        featureName = str(n)+'gram$gz'
        varcharLength = min((dlac.VARCHAR_WORD_LENGTH-(n-1))*n, 255)

        featureTableName = self.createFeatureTable(featureName, "VARCHAR(%d)"%varcharLength, 'INTEGER', tableName, valueFunc, correlField="INT(8)")

        mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting
        dlac.warn("extracting ngrams...")

        with gzip.open(gzCsv, 'rb') as gzFile:
            csv_reader = csv.reader(gzFile, delimiter=',', quotechar='"', escapechar='\\')

            msgs = 0#keeps track of the number of messages read
            seenMids = dict() #currently seen message ids
            freqs = dict() #holds frequency of n-grams [correl_id -> ngram -> freq]
            totalFreqs = dict() #total number of (non-distinct) n-grams seen for this correl field [correl_id -> totalGramFreq]

            #grab n-grams by messages for that cf:
            for record in csv_reader:
                # _warn(record)
                try:
                    message_id = record[idxIdField]
                    message = record[idxMsgField]
                    correl_id = record[-1].upper()
                except IndexError:
                    continue
                # _warn((message_id, message, correl_id))
                if not message_id in seenMids and message and correl_id:
                    msgs+=1
                    if msgs % dlac.PROGRESS_AFTER_ROWS == 0: #progress update
                        dlac.warn("Messages Read: %dk" % int(msgs/1000))
                    if msgs > 1000*2915:
                        break
                    message = tc.treatNewlines(message)
                    if self.use_unicode:
                        message = tc.removeNonUTF8(message)
                    else:
                        message = tc.removeNonAscii(message)
                    message = tc.shrinkSpace(message)

                    words = tokenizer.tokenize(message)
                    # _warn(words)
                    for ii in range(0,(len(words) - n)+1):
                        gram = ' '.join(words[ii:ii+n])
                        #truncate:
                        gram = gram[:varcharLength]
                        if lowercase_only: gram = gram.lower()

                        try:
                            freqs[correl_id][gram] += 1
                        except KeyError: #either correl_id does not exist or gram does not
                            try:
                                freqs[correl_id][gram] = 1 #gram did not exist but now its 1
                            except KeyError:
                                freqs[correl_id] = {gram:1} #correl_id did not exist but not its an initialized dictionary

                        try:
                            totalFreqs[correl_id] += 1.0
                        except KeyError:
                            totalFreqs[correl_id] = 1.0

                    seenMids[message_id] = 1

            #write n-grams to database (no need for "REPLACE" because we are creating the table)
            wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""
            rows = []
            for ii_correl, gramToFreq in freqs.items():
                if self.use_unicode:
                    ii_rows = [(ii_correl, k, v, valueFunc((v / totalFreqs[ii_correl])) ) for k, v in gramToFreq.items() if v >= min_freq] #adds group_norm and applies freq filter
                else:
                    ii_rows = [(ii_correl, k.encode('utf-8'), v, valueFunc((v / totalFreqs[ii_correl])) ) for k, v in gramToFreq.items() if v >= min_freq] #adds group_norm and applies freq filter
                rows.extend(ii_rows)
            dlac.warn( "Inserting %d rows..."%(len(rows),) )

            insert_idx_start = 0
            insert_idx_end = dlac.MYSQL_BATCH_INSERT_SIZE
            # write the rows in chunks
            while insert_idx_start < len(rows):
                insert_rows = rows[insert_idx_start:min(insert_idx_end, len(rows))]
                dlac.warn( "Inserting rows %d to %d..."%(insert_idx_start, insert_idx_end) )
                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, insert_rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                insert_idx_start += dlac.MYSQL_BATCH_INSERT_SIZE
                insert_idx_end += dlac.MYSQL_BATCH_INSERT_SIZE

        dlac.warn("Done Reading / Inserting.")

        # _warn("This tokenizer took %d seconds"%((datetime.utcnow()-t1).seconds,))

        dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
        mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys
        dlac.warn("Done\n")

        return featureTableName;


    def addLDAFeatTable(self, ldaMessageTable, tableName = None, valueFunc = lambda d: d):
        """???
        This assumes each row is a unique message, originally meant for twitter

        Parameters
        ----------
        ldaMessageTable : ?????
            ?????
        tableName : :obj:`str`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given

        Returns
        -------
        featureTableName : str
            Name of n-gram table: ?????
        """
        #CREATE TABLE:
        featureName =  'lda'+'_'+ldaMessageTable.split('$')[1]
        featureTableName = self.createFeatureTable(featureName, 'SMALLINT UNSIGNED', 'INTEGER', tableName, valueFunc)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        usql = """SELECT %s FROM %s GROUP BY %s""" % (
            self.correl_field, self.corptable, self.correl_field)
        msgs = 0#keeps track of the number of messages read
        cfRows = mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode)#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting
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
                    if msgs % dlac.PROGRESS_AFTER_ROWS == 0: #progress update
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
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

        dlac.warn("Done Reading / Inserting.")

        dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
        mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys
        dlac.warn("Done\n")
        return featureTableName;

    constParseMatchRe = re.compile(r'^\s*\([A-Z]')
    def addPhraseTable(self, tableName = None, valueFunc = lambda d: d, maxTaggedPhraseChars = 255):
        """Creates feature tuples (correl_field, feature, values) table where features are parsed phrases

        Parameters
        ----------
        tableName : :obj:`str`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given
        maxTaggedPhraseChars : :obj:`int`, optional
            ?????

        Returns
        -------
        phraseTableName : str
            Name of phrase table table: ?????
        """
        #CREATE TABLEs:
        maxPhraseChars = int(0.80*maxTaggedPhraseChars)
        taggedTableName = self.createFeatureTable('phrase_tagged', "VARCHAR(%d)"%maxTaggedPhraseChars, 'INTEGER', tableName, valueFunc)
        phraseTableName = self.createFeatureTable('phrase', "VARCHAR(%d)"%maxPhraseChars, 'INTEGER', tableName, valueFunc)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        parseTable = self.corptable+'_const'
        assert mm.tableExists(self.corpdb, self.dbCursor, parseTable, charset=self.encoding, use_unicode=self.use_unicode), "Need %s table to proceed with phrase featrue extraction " % parseTable
        usql = """SELECT %s FROM %s GROUP BY %s""" % (self.correl_field, parseTable, self.correl_field)
        msgs = 0#keeps track of the number of messages read
        cfRows = mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode)#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        mm.disableTableKeys(self.corpdb, self.dbCursor, taggedTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting
        mm.disableTableKeys(self.corpdb, self.dbCursor, phraseTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting
        for cfRow in cfRows:
            cf_id = cfRow[0]
            mids = set() #currently seen message ids
            freqsTagged = dict() #holds frequency of phrases
            freqsPhrases = dict() #holds frequency of phrases
            totalPhrases = 0

            #grab phrases by messages for that correl field:
            for messageRow in self.getMessagesForCorrelField(cf_id, messageTable = parseTable):
                message_id = messageRow[0]
                parse = messageRow[1]
                if not message_id in mids and parse:
                    msgs+=1
                    if msgs % dlac.PROGRESS_AFTER_ROWS == 0: #progress update
                        dlac.warn("Parsed Messages Read: %dk" % int(msgs/1000))
                    mids.add(message_id)

                    #find phrases and update freqs:
                    if (FeatureExtractor.constParseMatchRe.match(parse)):
                        (phrases, taggedPhrases) = self.findPhrasesInConstParse(parse)
                        for phrase in phrases:
                            if len(phrase) <= maxPhraseChars:
                                if phrase in freqsPhrases:
                                    freqsPhrases[phrase] += 1
                                else:
                                    freqsPhrases[phrase] = 1
                        for tPhrase in taggedPhrases:
                            if len(tPhrase) <= maxTaggedPhraseChars:
                                totalPhrases +=1
                                if tPhrase in freqsTagged:
                                    freqsTagged[tPhrase] += 1
                                else:
                                    freqsTagged[tPhrase] = 1
                    else:
                        dlac.warn("*Doesn't appear to be a parse:\n%s\n*Perhaps a problem with the parser?\n" % parse)


            #write phrases to database (no need for "REPLACE" because we are creating the table)
            totalPhrases = float(totalPhrases) #to avoid casting each time below

            wsql = """INSERT INTO """+phraseTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
            phraseRows = [(k, v, valueFunc((v / totalPhrases))) for k, v in freqsPhrases.items()] #adds group_norm and applies freq filter
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, phraseRows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

            wsql = """INSERT INTO """+taggedTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
            taggedRows = [(k, v, valueFunc((v / totalPhrases))) for k, v in freqsTagged.items()] #adds group_norm and applies freq filter
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, taggedRows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

        dlac.warn("Done Reading / Inserting.")

        dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
        mm.enableTableKeys(self.corpdb, self.dbCursor, taggedTableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys
        mm.enableTableKeys(self.corpdb, self.dbCursor, phraseTableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys
        dlac.warn("Done\n")
        return phraseTableName;

    def findPhrasesInConstParse(self, parse):
        """Traverses a constituent parse tree to pull out all phrases

        Parameters
        ----------
        parse : :obj:`?????`, optional
            ?????

        Returns
        -------
        (strings, tagged) : (list, list)
            ?????
        """
        tries = 0
        trees = []
        while (True):
            try:
                trees = [ParentedTree(parse)]
                break
            except ValueError as err:
                dlac.warn("*ValueError when trying to create tree:\n %s\n %s\n adjusting parens and trying again"
                      % (err, parse))
                tries +=1
                if tries < 2:
                    parse = parse[:-1]
                elif tries < 3:
                    parse = parse+'))'
                elif tries < 8:
                    parse = parse+')'
                else:
                    dlac.warn("\n done trying, moving on\n")
                    return ([], [])

        tagged = []
        strings = []
        while trees:
            t = trees.pop()
            tagString = ' '.join(str(t).split())
            tagged.append(tagString)
            leaves = t.leaves()
            if not t.parent or not leaves == t.parent.leaves:
                strings.append(' '.join(leaves))
            if len(leaves) > 1:
                for child in t:
                    if not isinstance(child, str):
                        trees.append(child)

        #pprint((strings, tagged))
        return (strings, tagged)


    def addPNamesTable(self, nameLex, languageLex, tableName = None, valueFunc = lambda d: d, lastNameCat = 'LAST', firstNameCats=['FIRST-FEMALE', 'FIRST-MALE']):
        """Creates feature tuples (correl_field, feature, values) table where features are People's Names

        Parameters
        ----------
        nameLex : str
            ?????
        languageLex : ?????
            ?????
        tableName : :obj:`?????`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given
        lastNameCat : :obj:`str`, optional
            ?????
        firstNameCats : :obj:`list`, optional
            ?????


        Returns
        -------
        (strings, tagged) : (list, list)
            ?????
        """
        ##NOTE: correl_field should have an index for this to be quick
        tokenizer = Tokenizer(preserve_case=True, use_unicode=self.use_unicode)

        #setup lexicons
        lastNames = frozenset([name.title() for name in nameLex[lastNameCat]])
        firstNames = frozenset([name.title() for c in firstNameCats for name in nameLex[c]])
        langWords = frozenset([word for c in languageLex.keys() for word in languageLex[c]])

        #CREATE TABLE:
        featureName = 'PNames'
        varcharLength = min((dlac.VARCHAR_WORD_LENGTH-(3))*4, 255)
        featureTableName = self.createFeatureTable(featureName, "VARCHAR(%d)"%varcharLength, 'INTEGER', tableName, valueFunc)


        #SELECT / LOOP ON CORREL FIELD FIRST:
        usql = """SELECT %s FROM %s GROUP BY %s""" % (
            self.correl_field, self.corptable, self.correl_field)
        msgs = 0 # keeps track of the number of messages read
        cfRows = FeatureExtractor.noneToNull(mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode))#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS: mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting
        for cfRow in cfRows:
            cf_id = cfRow[0]

            mids = set() #currently seen message ids
            freqs = dict() #holds frequency of n-grams
            totalGrams = 0 #total number of (non-distinct) n-grams seen for this user

            #grab n-grams by messages for that cf:
            for messageRow in self.getMessagesForCorrelField(cf_id, warnMsg = False):
                message_id = messageRow[0]
                message = messageRow[1]
                if not message_id in mids and message:
                    msgs+=1
                    if msgs % dlac.PROGRESS_AFTER_ROWS == 0: #progress update
                        dlac.warn("Messages Read: %dk" % int(msgs/1000))
                    message = tc.treatNewlines(message)
                    if self.use_unicode:
                        message = tc.removeNonUTF8(message)
                    else:
                        message = tc.removeNonAscii(message)
                    message = tc.shrinkSpace(message)


                    #words = message.split()
                    words = tokenizer.tokenize(message)

                    for n in range(2, 5):
                        for i in range(0,(len(words) - n)+1):
                            gram = None
                            if n == 2: totalGrams += 1 #only increment on 2grams
                            gramWords = words[i:i+n]

                            #check if first / last are in dictionaries (title caps)
                            if i > 0 and gramWords[0] in firstNames and gramWords[-1] in lastNames:
                                #> 0 eleminates first words being capitalized (must go past next filter)
                                #at least one word should NOT be in the lowercase dictionary:
                                keep = False
                                for word in gramWords:
                                    if word.lower() not in langWords:
                                        keep = True
                                        break
                                if keep:
                                    gram = ' '.join(gramWords)

                            else:
                                if gramWords[0].lower().title() in firstNames and gramWords[-1].lower().title() in lastNames:
                                    #need to make sure other dictionary words aren't present.
                                    keep = True
                                    otherWords = set()
                                    for word in gramWords:
                                        if word.lower() in langWords and word.lower() not in otherWords:
                                            keep = False
                                            break
                                        otherWords.add(word.lower())
                                    if keep:
                                        if not (gramWords[0] == gramWords[1] and (len(gramWords) > 2 and gramWords[1] == gramWords[2])):
                                            gram = ' '.join(gramWords)

                            if gram:
                            #truncate:
                                gram = gram[:varcharLength]

                                try:
                                    freqs[gram] += 1
                                except KeyError:
                                    freqs[gram] = 1

                    mids.add(message_id)

            #write n-grams to database (no need for "REPLACE" because we are creating the table)
            if totalGrams:
                wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
                totalGrams = float(totalGrams) # to avoid casting each time below
                if self.use_unicode:
                    rows = [(k, v, valueFunc((v / totalGrams))) for k, v in freqs.items()] #adds group_norm and applies freq filter
                else:
                    rows = [(k.encode('utf-8'), v, valueFunc((v / totalGrams))) for k, v in freqs.items()] #adds group_norm and applies freq filter

                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

        dlac.warn("Done Reading / Inserting.")

        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys
        dlac.warn("Done\n")
        return featureTableName

    def addFleschKincaidTable(self, tableName = None, valueFunc = lambda d: d, removeXML = True, removeURL = True):
        """Creates feature tuples (correl_field, feature, values) table where features are flesch-kincaid scores.

        Parameters
        ----------
        tableName : :obj:`str`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given
        removeXML : :obj:`boolean`, optional
            ?????
        removeURL : :obj:`boolean`, optional
            ?????


        Returns
        -------
        featureTableName : str
            Name of Flesch Kincaid table: feat$flkin$corptable$correl_field%transform
        """

        ##NOTE: correl_field should have an index for this to be quick
        fk_score = textstat.flesch_kincaid_grade

        #CREATE TABLE:
        featureName = 'flkin'
        featureTableName = self.createFeatureTable(featureName, "VARCHAR(8)", 'FLOAT', tableName, valueFunc)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        usql = """SELECT %s FROM %s GROUP BY %s""" % (
            self.correl_field, self.corptable, self.correl_field)
        msgs = 0 # keeps track of the number of messages read
        cfRows = FeatureExtractor.noneToNull(mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode))#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS: mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting

        for cfRow in cfRows:
            cf_id = cfRow[0]

            mids = set() #currently seen message ids
            scores = list()

            #grab n-grams by messages for that cf:
            for messageRow in self.getMessagesForCorrelField(cf_id, warnMsg = False):
                message_id = messageRow[0]
                message = messageRow[1]
                if not message_id in mids and message:
                    msgs+=1
                    if msgs % dlac.PROGRESS_AFTER_ROWS == 0: #progress update
                        dlac.warn("Messages Read: %dk" % int(msgs/1000))

                    #TODO: replace <br />s with a period.
                    if removeXML: message = FeatureExtractor.removeXML(message)
                    if removeURL: message = FeatureExtractor.removeURL(message)
                    message = FeatureExtractor.shortenDots(message)
                    try:
                        s = min(fk_score(message), 20)#maximize at "20th" grade in case bug
                        scores.append(s)
                        mids.add(message_id)
                    except ZeroDivisionError:
                        dlac.warn("unable to get Flesch-Kincaid score for: %s\n  ...skipping..." % message)



            if mids:
                avg_score = mean(scores)
                wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
                rows = [("m_fk_score", avg_score, valueFunc(avg_score))] #adds group_norm and applies freq filter
                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

        dlac.warn("Done Reading / Inserting.")

        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys
        dlac.warn("Done\n")
        return featureTableName



    ##HELPER METHODS##


    def getCorrelFieldType(self, correlField):
        """Returns the type of correlField

        Parameters
        ----------
        correlField : str
            Correlation Field (AKA Group Field): The field which features are aggregated over
        """
        if correlField == 'state':
            return 'char(2)'
        return None

    def createFeatureTable(self, featureName, featureType = 'VARCHAR(64)', valueType = 'INTEGER', tableName = None, valueFunc = None, correlField=None, extension = None):
        """Creates a feature table based on self data and feature name

        Parameters
        ----------
        featureName : str
            Type of feature table (ex: 1gram, 1to3gram, cat_LIWC).
        featureType : :obj:`str`, optional
            MySQL type of feature.
        valueType : :obj:`str`, optional
            MySQL type of value.
        tableName : :obj:`str`, optional
            Name of table to be created. If not supplied the name will be automatically generated.
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given
        correlField : :obj:`str`, optional
            Correlation Field (AKA Group Field): The field which features are aggregated over
        extension : :obj:`str`, optional
            ?????

        """
        #create table name
        if not tableName:
            tableName = 'feat$'+featureName+'$'+self.corptable+'$'+self.correl_field
            if valueFunc:
                tableName += '$' + str(16)+'to'+"%d"%round(valueFunc(16))
            if extension:
                tableName += '$' + extension

        #find correl_field type:
        sql = """SELECT column_type FROM information_schema.columns WHERE table_schema='%s' AND table_name='%s' AND column_name='%s'""" % (
            self.corpdb, self.corptable, self.correl_field)
        try:
            correlField = self.getCorrelFieldType(self.correl_field) if not correlField else correlField
            correl_fieldType = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)[0][0] if not correlField else correlField
        except IndexError:
            dlac.warn("Your message table '%s' (or the group field, '%s') probably doesn't exist (or the group field)!" %(self.corptable, self.correl_field))
            raise IndexError("Your message table '%s' probably doesn't exist!" % self.corptable)

        #create sql
        drop = """DROP TABLE IF EXISTS %s""" % tableName
        sql = """CREATE TABLE %s (id BIGINT(16) UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                 group_id %s, feat %s CHARACTER SET %s COLLATE %s, value %s, group_norm DOUBLE,
                 KEY `correl_field` (`group_id`), KEY `feature` (`feat`))
                 CHARACTER SET %s COLLATE %s ENGINE=%s""" %(tableName, correl_fieldType, featureType, self.encoding, dlac.DEF_COLLATIONS[self.encoding.lower()], valueType, self.encoding, dlac.DEF_COLLATIONS[self.encoding.lower()], dlac.DEF_MYSQL_ENGINE)

        #run sql
        mm.execute(self.corpdb, self.dbCursor, drop, charset=self.encoding, use_unicode=self.use_unicode)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

        return tableName

    def createLexFeatTable(self, lexiconTableName, lexKeys, isWeighted=False, tableName = None, valueFunc = None, correlField=None, extension = None):
        """
        Creates a feature table of the form lex$featureType$messageTable$groupID$valueFunc$ext.
        This table is used when printing topic tagclouds and looks at the corpus the lexicon is applied to
        rather than relying on the posteriors from the model to dictate which words to display for a topic.

        Parameters
        ----------
        lexiconTableName : str
            ?????
        lexKeys : list
            ?????
        isWeighted : boolean
            ?????
        tableName : str
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given
        correlField : :obj:`str`, optional
            Correlation Field (AKA Group Field): The field which features are aggregated over
        extension : :obj:`str`, optional
            ?????

        Returns
        -------
        tableName : str
            Name of created feature table: lex%cat_lexTable%corptable$correl_field

        """
        #create table name
        if not tableName:
            if isWeighted:
                lexiconTableName += '_w'
            tableName = 'lex$cat_' + lexiconTableName + '$' + self.corptable + '$' + self.correl_field
            if valueFunc:
                tableName += '$' + str(16) + 'to' + "%d" % round(valueFunc(16))
            if extension:
                tableName += '$' + extension

        #first create the table:
        enumCats = "'" + "', '".join([k.upper().replace("'", "\\'") for k in lexKeys]) + "'"
        drop = """DROP TABLE IF EXISTS """ + tableName
        sql = """CREATE TABLE IF NOT EXISTS %s (id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                term VARCHAR(140), category ENUM(%s), weight DOUBLE, INDEX(term), INDEX(category)) CHARACTER SET %s COLLATE %s ENGINE=%s""" % (tableName, enumCats, self.encoding, dlac.DEF_COLLATIONS[self.encoding.lower()], dlac.DEF_MYSQL_ENGINE)
        #run sql
        mm.execute(self.corpdb, self.dbCursor, drop, charset=self.encoding, use_unicode=self.use_unicode)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

        return tableName

    def addCorpLexTable(self, lexiconTableName, lowercase_only=dlac.LOWERCASE_ONLY, tableName=None, valueFunc = lambda x: float(x), isWeighted=False, featValueFunc=lambda d: float(d)):
        """?????

        Parameters
        ----------
        lexiconTableName : str
            ?????
        lowercase_only : boolean
            use only lowercase charngrams if True
        tableName : :obj:`str`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given
        isWeighted : :obj:`boolean`, optional
            Is the lexcion weighted?
        featValueFunc : :obj:`lambda`, optional
            ?????

        Returns
        -------
        tableName : str
            Name of created feature table: lex%cat_lexTable%corptable$correl_field

        """

        feat_cat_weight = dict()
        sql = "SELECT * FROM %s.%s"%(self.lexicondb, lexiconTableName)
        rows = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
        categories = set()
        lexiconHasWildCard = False
        warnedAboutWeights = False

        for row in rows:
            term = row[1].strip()
            category = row[2].strip()

            if term and category:
                weight = 1
                if isWeighted:
                    try:
                        weight = row[3]
                    except IndexError:
                        print('\nERROR: The lexicon you specified is probably not weighted, or there is a problem in the lexicon itself (Check DB)')
                        sys.exit(2)
                elif len(row) == 4 and not warnedAboutWeights:
                    dlac.warn("""###################################################################
  WARNING: The lexicon you specified has weights, but you didn't
  specify --weighted_lexicon
###################################################################""")
                    sys.exit(2)
                    warnedAboutWeights = True
                if lowercase_only: term = term.lower()
                if term == '_intercept':
                    dlac.warn("Intercept detected %f [category: %s]" % (weight,category))
                    _intercepts[category] = weight
                if term[-1] == '*':
                    lexiconHasWildCard = True
                feat_cat_weight[term] = feat_cat_weight.get(term,{})
                feat_cat_weight[term][category] = weight
                categories.add(category)

        wordTable = self.getWordTable()
        dlac.warn("WORD TABLE %s"%(wordTable,))

        if not tableName:
            tableName = self.createLexFeatTable(lexiconTableName=lexiconTableName, lexKeys=categories,  isWeighted=isWeighted, tableName=tableName, valueFunc=valueFunc, correlField=None, extension=None)

        rowsToInsert = []

        isql = "INSERT IGNORE INTO "+tableName+" (term, category, weight) values (%s, %s, %s)"
        reporting_percent = 0.01
        reporting_int = max(floor(reporting_percent * len(feat_cat_weight)), 1)
        featIdCounter = 0

        for feat in feat_cat_weight:
            sql = """SELECT feat, avg(group_norm) FROM %s WHERE feat LIKE "%s" """ % (wordTable, mm.MySQLdb.escape_string(feat))
            attributeRows = mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode)[0]
            if attributeRows[0]:
                rows = [(feat, topic, str(feat_cat_weight[feat][topic]*attributeRows[1])) for topic in feat_cat_weight[feat]]

            rowsToInsert.extend(rows)

            if len(rowsToInsert) > dlac.MYSQL_BATCH_INSERT_SIZE:
                mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                rowsToInsert = []
            featIdCounter += 1
            if featIdCounter % reporting_int == 0:
                dlac.warn("%d out of %d features processed; %2.2f complete"%(featIdCounter, len(feat_cat_weight), float(featIdCounter)/len(feat_cat_weight)))

        if len(rowsToInsert) > 0:
            mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
            rowsToInsert = []
            dlac.warn("%d out of %d features processed; %2.2f complete"%(featIdCounter, len(feat_cat_weight), float(featIdCounter)/len(feat_cat_weight)))

#        mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

        return tableName

    def addLexiconFeat(self, lexiconTableName, lowercase_only=dlac.LOWERCASE_ONLY, tableName=None, valueFunc = lambda x: float(x), isWeighted=False, featValueFunc=lambda d: float(d), extension=None):
        """Creates a feature table given a 1gram feature table name, a lexicon table / database name

        Parameters
        ----------
        lexiconTableName : str
            ?????
        lowercase_only : boolean
            use only lowercase charngrams if True
        tableName : :obj:`str`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given
        isWeighted : :obj:`boolean`, optional
            Is the lexcion weighted?
        featValueFunc : :obj:`lambda`, optional
            ?????

        Returns
        -------
        tableName : str
            Name of created feature table: feat%cat_lexTable%corptable$correl_field

        """
        PR = pprint #debug

        _intercepts = {}
        #1. word -> set(category) dict
        #2. Get length for varchar column
        feat_cat_weight = dict()
        sql = "SELECT * FROM %s.%s"%(self.lexicondb, lexiconTableName)
        rows = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
        categories = set()
        lexiconHasWildCard = False
        warnedAboutWeights = False
        max_category_string_length = -1
        for row in rows:
            #e.g. (2, "bored", "E-")
            #OR   (2, "bored", "E-", "1")
            term = row[1].strip()
            category = row[2].strip()
            if term and category:
                weight = 1
                if isWeighted:
                    try:
                        weight = row[3]
                    except IndexError:
                        print('\nERROR: The lexicon you specified is probably not weighted, or there is a problem in the lexicon itself (Check DB)')
                        sys.exit(2)
                elif len(row) == 4 and not warnedAboutWeights:
                    dlac.warn("""###################################################################
  WARNING: The lexicon you specified has weights, but you didn't
  specify --weighted_lexicon so the weights won't be used
###################################################################""")
                    warnedAboutWeights = True
                if lowercase_only: term = term.lower()
                if term == '_intercept':
                    dlac.warn("Intercept detected %f [category: %s]" % (weight,category))
                    _intercepts[category] = weight
                if term[-1] == '*':
                    lexiconHasWildCard = True
                feat_cat_weight[term] = feat_cat_weight.get(term,{})
                feat_cat_weight[term][category] = weight
                categories.add(category)
                if len(category) > max_category_string_length:
                    max_category_string_length = len(category)

        #3. create new Feature Table
        if isWeighted:
            lexiconTableName += "_w"
        if featValueFunc(16) != 16:
            lexiconTableName += "_16to"+str(int(featValueFunc(16)))


        tableName = self.createFeatureTable("cat_%s"%lexiconTableName, 'VARCHAR(%d)'%max_category_string_length, 'INTEGER', tableName, valueFunc, extension=extension)


        #4. grab all distinct group ids
        wordTable = self.getWordTable()
        dlac.warn("WORD TABLE %s"%(wordTable,))

        assert mm.tableExists(self.corpdb, self.dbCursor, wordTable, charset=self.encoding, use_unicode=self.use_unicode), "Need to create word table to extract the lexicon: %s" % wordTable
        sql = "SELECT DISTINCT group_id FROM %s" % wordTable
        groupIdRows = mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode)

        #5. disable keys on that table if we have too many entries
        #if (len(categories)* len(groupIdRows)) < dlac.MAX_TO_DISABLE_KEYS:
        mm.disableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode) #for faster, when enough space for repair by sorting

        #6. iterate through source feature table by group_id (fixed, column name will always be group_id)
        rowsToInsert = []

        isql = "INSERT IGNORE INTO "+tableName+" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"
        reporting_percent = 0.01
        reporting_int = max(floor(reporting_percent * len(groupIdRows)), 1)
        groupIdCounter = 0
        for groupIdRow in groupIdRows:
            groupId = groupIdRow[0]

            #i. create the group_id category counts & keep track of how many features they have total
            cat_to_summed_value = dict()
            cat_to_function_summed_weight = dict()
            cat_to_function_summed_weight_gn = {}
            sql = ''
            if isinstance(groupId, str):
                sql = "SELECT group_id, feat, value, group_norm FROM %s WHERE group_id LIKE '%s'"%(wordTable, groupId)
            else:
                sql = "SELECT group_id, feat, value, group_norm FROM %s WHERE group_id = %d"%(wordTable, groupId)

            try:
                attributeRows = mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode)
            except:
                print(groupId)
                sys.exit()

            totalFeatCountForThisGroupId = 0

            totalFunctionSumForThisGroupId = float(0.0)
            totalWordsInLexForThisGroupId = float(0.0)
            for (gid, feat, value, group_norm) in attributeRows:
                #e.g. (69L, 8476L, 'spent', 1L, 0.00943396226415094, None),
                cat_to_weight = dict()#dictionary holding all categories, weights that feat is a part of
                if not feat: continue
                if lowercase_only: feat = feat.lower()
                totalFeatCountForThisGroupId += value
                totalFunctionSumForThisGroupId += featValueFunc(value)

                if feat in feat_cat_weight:
                    cat_to_weight.update(feat_cat_weight[feat]) #*value) #???
                    #totalWordsInLexForThisGroupId += featValueFunc(value)
                if lexiconHasWildCard: #check wildcard matches
                    for endI in range(3, len(feat)+1):
                        featWild = feat[0:endI]+'*'
                        if featWild in feat_cat_weight:
                            cat_to_weight = dlac.unionDictsMaxOnCollision(cat_to_weight, feat_cat_weight[featWild])
                #update all cats:
                for category in cat_to_weight:
                    try:
                        cat_to_summed_value[category] += value
                        cat_to_function_summed_weight[category] += cat_to_weight[category] * featValueFunc(value)
                        cat_to_function_summed_weight_gn[category] += cat_to_weight[category] * featValueFunc(group_norm)
                    except KeyError:
                        cat_to_summed_value[category] = value
                        cat_to_function_summed_weight[category] = cat_to_weight[category] * featValueFunc(value)
                        cat_to_function_summed_weight_gn[category] = cat_to_weight[category] * featValueFunc(group_norm)

            # print gid
            # pprint(cat_to_function_summed_weight) #debug
            # ii. Calculate the group norms (the percentage of the gid's words observed in each category transformed by valueFunc (e.g. sqrt))
            # totalFeatCountForThisGroupId = float(totalFeatCountForThisGroupId) #to avoid casting each time below

            # Using the value and applying the featValueFunction to the value and the UWT separately
            # rows = [(gid, k.encode('utf-8'), cat_to_summed_value[k], valueFunc(_intercepts.get(k,0)+(v / totalFunctionSumForThisGroupId))) for k, v in cat_to_function_summed_weight.iteritems()]

            # Applying the featValueFunction to the group_norm,
            if self.use_unicode:
                rows = [(gid, k, cat_to_summed_value[k], valueFunc(_intercepts.get(k,0)+v)) for k, v in cat_to_function_summed_weight_gn.items()]
            else:
                rows = [(gid, k.encode('utf-8'), cat_to_summed_value[k], valueFunc(_intercepts.get(k,0)+v)) for k, v in cat_to_function_summed_weight_gn.items()]


            # iii. Insert data into new feautre table
            # Add new data to rows to be inserted into the database
            # Check if size is big enough for a batch insertion (10,000?), if so insert and clear list
            rowsToInsert.extend(rows)
            if len(rowsToInsert) > dlac.MYSQL_BATCH_INSERT_SIZE:
                mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                rowsToInsert = []
            groupIdCounter += 1
            if groupIdCounter % reporting_int == 0:
                dlac.warn("%d out of %d group Id's processed; %2.2f complete"%(groupIdCounter, len(groupIdRows), float(groupIdCounter)/len(groupIdRows)))

        #7. if any data in the data_to_insert rows, insert the data and clear the list
        if len(rowsToInsert) > 0:
            mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
            rowsToInsert = []
            dlac.warn("%d out of %d group Id's processed; %2.2f complete"%(groupIdCounter, len(groupIdRows), float(groupIdCounter)/len(groupIdRows)))

        #8. enable keys on the new feature table
        #if (len(categories)* len(groupIdRows)) < dlac.MAX_TO_DISABLE_KEYS:
        mm.enableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys

        #9. exit with success, return the newly created feature table
        return tableName

    def addWNNoPosFeat(self, tableName=None, valueFunc = lambda x: float(x), featValueFunc=lambda d: float(d)):
        """Creates a wordnet concept feature table (based on words without pos tags) given a 1gram feature table name

        Parameters
        ----------
        tableName : :obj:`str`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given
        featValueFunc : :obj:`lambda`, optional
            ?????

        Returns
        -------
        tableName : str
            Name of created feature table: feat%wn_nopos%corptable$correl_field

        """

        #1. Load WordNet Object

        #2. create new Feature Table
        shortName = 'wn_nopos'
        if featValueFunc(16) != 16:
            shortName += "_16to"+str(int(featValueFunc(16)))
        tableName = self.createFeatureTable(shortName, 'VARCHAR(48)', 'DOUBLE', tableName, valueFunc)

        #3. grab all distinct group ids
        wordTable = self.getWordTable()
        dlac.warn("WORD TABLE %s"%(wordTable,))
        assert mm.tableExists(self.corpdb, self.dbCursor, wordTable, charset=self.encoding, use_unicode=self.use_unicode), "Need to create word table to apply groupThresh: %s" % wordTable
        sql = "SELECT DISTINCT group_id FROM %s"%wordTable
        groupIdRows = mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode)

        #4. disable keys on that table if we have too many entries
        mm.disableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting

        #5. iterate through source feature table by group_id (fixed, column name will always be group_id)
        rowsToInsert = []

        isql = "INSERT IGNORE INTO "+tableName+" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"
        reporting_percent = 0.01
        reporting_int = max(floor(reporting_percent * len(groupIdRows)), 1)
        groupIdCounter = 0
        #stripsynset = re.compile(r'')
        for groupIdRow in groupIdRows:
            groupId = groupIdRow[0]

            #5.a. create the group_id concept counts & keep track of how many features they have total
            cncpt_to_summed_value = dict()
            cncpt_to_function_summed_value = dict()
            sql = ''
            if isinstance(groupId, str):
                sql = "SELECT group_id, feat, value, group_norm FROM %s WHERE group_id LIKE '%s'"%(wordTable, groupId)
            else:
                sql = "SELECT group_id, feat, value, group_norm FROM %s WHERE group_id = %d"%(wordTable, groupId)
            attributeRows = mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode)

            totalFunctionSumForThisGroupId = float(0.0)
            for (gid, feat, value, group_norm) in attributeRows:
                #e.g. (69L, 8476L, 'spent', 1L, 0.00943396226415094, None),

                #traverse ontology
                synsets = wn.synsets(feat, pos=wn.NOUN)
                if len(synsets) > 0:
                    totalFunctionSumForThisGroupId += featValueFunc(value)
                    funcProbs = [float(featValueFunc(value) / float(len(synsets)))] * len(synsets) #split value across all concepts
                    probs = [float(value / float(len(synsets)))] * len(synsets) #split value across all concepts
                    while len(synsets) > 0:
                        currentSynset = synsets.pop()
                        currentProb = probs.pop()
                        currentFuncProb = funcProbs.pop()
                        currentSynsetStr = str(currentSynset).split("(")[1].strip(")'")
                        if currentSynsetStr in cncpt_to_summed_value:
                            cncpt_to_summed_value[currentSynsetStr] += currentProb
                            cncpt_to_function_summed_value[currentSynsetStr] += currentFuncProb
                        else:
                            cncpt_to_summed_value[currentSynsetStr] = currentProb
                            cncpt_to_function_summed_value[currentSynsetStr] = currentFuncProb
                        hypes = []
                        hypes.extend(currentSynset.hypernyms())
                        hypes.extend(currentSynset.instance_hypernyms())
                        if hypes:
                            synsets.extend(hypes)
                            probs.extend([float(currentProb) / len(hypes)] * len(hypes))
                            funcProbs.extend([float(currentFuncProb) / len(hypes)] * len(hypes))
                        elif "entity.n.01" not in str(currentSynset):
                            print(currentSynset)

            #5.b.. Insert data into new feautre table
            # Add new data to rows to be inserted into the database
            #pprint(cncpt_to_function_summed_value)
            #print totalFunctionSumForThisGroupId
            if self.use_unicode:
                rows = [(gid, k, cncpt_to_summed_value[k], valueFunc((v / totalFunctionSumForThisGroupId))) for k, v in cncpt_to_function_summed_value.items()]
            else:
                rows = [(gid, k.encode('utf-8'), cncpt_to_summed_value[k], valueFunc((v / totalFunctionSumForThisGroupId))) for k, v in cncpt_to_function_summed_value.items()]
            rowsToInsert.extend(rows)
            if len(rowsToInsert) > dlac.MYSQL_BATCH_INSERT_SIZE:
                mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                rowsToInsert = []
            groupIdCounter += 1
            if groupIdCounter % reporting_int == 0:
                dlac.warn("%d out of %d group Id's processed; %2.2f complete"%(groupIdCounter, len(groupIdRows), float(groupIdCounter)/len(groupIdRows)))

        #6. if any data in the data_to_insert rows, insert the data and clear the list
        if len(rowsToInsert) > 0:
            mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
            rowsToInsert = []

        #7. enable keys on the new feature table
        mm.enableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys

        #8. exit with success, return the newly created feature table
        return tableName

    def addWNPosFeat(self, tableName=None, pos_table = None, valueFunc = lambda x: float(x), featValueFunc=lambda d: float(d)):
        """Creates a wordnet concept feature table (based on words with pos tags) given a POS feature table name

        Parameters
        ----------
        tableName : :obj:`str`, optional
            ?????
        pos_table : :obj:`str`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given
        featValueFunc : :obj:`lambda`, optional
            ?????

        Returns
        -------
        tableName : str
            Name of created feature table: feat%wn_pos%corptable$correl_field

        """
        #1. Load WordNet Object

        #2. create new Feature Table
        shortName = 'wn_pos'
        if featValueFunc(16) != 16:
            shortName += "_16to"+str(int(featValueFunc(16)))
        tableName = self.createFeatureTable(shortName, 'VARCHAR(48)', 'DOUBLE', tableName, valueFunc)

        #3. grab all distinct group ids
        wordTable = self.getWordTable()
        dlac.warn("WORD TABLE %s"%(wordTable,))
        assert mm.tableExists(self.corpdb, self.dbCursor, wordTable, charset=self.encoding, use_unicode=self.use_unicode), "Need to create 1gram 16to16 table to apply groupThresh: %s" % wordTable

        #3.2 check that the POS table exists
        if not pos_table:
            pos_table = "feat$1gram_pos$%s$%s$16to16" %(self.corptable, self.correl_field)
        dlac.warn("POS TABLE: %s"%(pos_table,))
        assert mm.tableExists(self.corpdb, self.dbCursor, pos_table, charset=self.encoding, use_unicode=self.use_unicode), "Need to create POS table to apply functionality: %s" % pos_table
        sql = "SELECT DISTINCT group_id FROM %s"%pos_table
        groupIdRows = mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode)

        #4. disable keys on that table if we have too many entries
        mm.disableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting

        #5. iterate through source feature table by group_id (fixed, column name will always be group_id)
        rowsToInsert = []

        isql = "INSERT IGNORE INTO "+tableName+" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"
        reporting_percent = 0.01
        reporting_int = max(floor(reporting_percent * len(groupIdRows)), 1)
        groupIdCounter = 0
        #stripsynset = re.compile(r'')
        for groupIdRow in groupIdRows:
            groupId = groupIdRow[0]

            #5.a. create the group_id concept counts & keep track of how many features they have total
            cncpt_to_summed_value = dict()
            cncpt_to_function_summed_value = dict()
            sql = ''
            if isinstance(groupId, str):
                sql = "SELECT group_id, feat, value, group_norm FROM %s WHERE group_id LIKE '%s'"%(pos_table, groupId)
            else:
                sql = "SELECT group_id, feat, value, group_norm FROM %s WHERE group_id = %d"%(pos_table, groupId)
            attributeRows = mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode)

            totalFunctionSumForThisGroupId = float(0.0)
            for (gid, feat, value, group_norm) in attributeRows:
                #e.g. (69L, 8476L, 'spent/V', 1L, 0.00943396226415094, None),

                #traverse ontology
                splitlist = feat.split('/')
                word, POS = ('/'.join(splitlist[:-1]),splitlist[-1])

                if POS != 'NN':
                    continue

                synsets = wn.synsets(word, pos=wn.NOUN)
                # Maarten: remove?
                # feat = word

                if len(synsets) > 0:
                    totalFunctionSumForThisGroupId += featValueFunc(value)
                    funcProbs = [float(featValueFunc(value) / float(len(synsets)))] * len(synsets) #split value across all concepts
                    probs = [float(value / float(len(synsets)))] * len(synsets) #split value across all concepts
                    while len(synsets) > 0:
                        currentSynset = synsets.pop()
                        currentProb = probs.pop()
                        currentFuncProb = funcProbs.pop()
                        currentSynsetStr = str(currentSynset).split("(")[1].strip(")'")
                        if currentSynsetStr in cncpt_to_summed_value:
                            cncpt_to_summed_value[currentSynsetStr] += currentProb
                            cncpt_to_function_summed_value[currentSynsetStr] += currentFuncProb
                        else:
                            cncpt_to_summed_value[currentSynsetStr] = currentProb
                            cncpt_to_function_summed_value[currentSynsetStr] = currentFuncProb
                        hypes = []
                        hypes.extend(currentSynset.hypernyms())
                        hypes.extend(currentSynset.instance_hypernyms())
                        if hypes:
                            synsets.extend(hypes)
                            probs.extend([float(currentProb) / len(hypes)] * len(hypes))
                            funcProbs.extend([float(currentFuncProb) / len(hypes)] * len(hypes))
                        elif "entity.n.01" not in str(currentSynset):
                            print(currentSynset)

            #5.b.. Insert data into new feautre table
            # Add new data to rows to be inserted into the database
            #pprint(cncpt_to_function_summed_value)
            #print totalFunctionSumForThisGroupId
            if self.use_unicode:
                rows = [(gid, k, cncpt_to_summed_value[k], valueFunc((v / totalFunctionSumForThisGroupId))) for k, v in cncpt_to_function_summed_value.items()]
            else:
                rows = [(gid, k.encode('utf-8'), cncpt_to_summed_value[k], valueFunc((v / totalFunctionSumForThisGroupId))) for k, v in cncpt_to_function_summed_value.items()]
            rowsToInsert.extend(rows)

            if len(rowsToInsert) > dlac.MYSQL_BATCH_INSERT_SIZE:
                mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                rowsToInsert = []
            groupIdCounter += 1
            if groupIdCounter % reporting_int == 0:
                dlac.warn("%d out of %d group Id's processed; %2.2f complete"%(groupIdCounter, len(groupIdRows), float(groupIdCounter)/len(groupIdRows)))

        #6. if any data in the data_to_insert rows, insert the data and clear the list
        if len(rowsToInsert) > 0:
            mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
            rowsToInsert = []

        #7. enable keys on the new feature table
        mm.enableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys

        #8. exit with success, return the newly created feature table
        return tableName


    def addPosTable(self, tableName = None, valueFunc = lambda d: d, keep_words = False):
        """Creates feature tuples (correl_field, feature, values) table where features are parts of speech

        Parameters
        ----------
        tableName : :obj:`str`, optional
            ?????
        pos_table : :obj:`str`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given
        featValueFunc : :obj:`lambda`, optional
            ?????

        Returns
        -------
        posFeatTableName : str
            Name of created feature table: feat%pos%corptable$correl_field or feat%1gram_pos%corptable$correl_field

        """
        # keep_words means it's not going to just count "JJ" or "PP", but "nice/JJ" etc.
        #CREATE TABLEs:

        alter_table = False
        if keep_words:
            min_varchar_length = 64
            posFeatTableName = self.createFeatureTable('1gram_pos', "VARCHAR(%s)" % min_varchar_length, 'INTEGER', tableName, valueFunc)
        else:
            min_varchar_length = 12
            posFeatTableName = self.createFeatureTable('pos', "VARCHAR(%s)" % min_varchar_length, 'INTEGER', tableName, valueFunc)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        posMessageTable = self.corptable+'_pos'
        assert mm.tableExists(self.corpdb, self.dbCursor, posMessageTable, charset=self.encoding, use_unicode=self.use_unicode), "Need %s table to proceed with pos featrue extraction " % posMessageTable
        usql = """SELECT %s FROM %s GROUP BY %s""" % (self.correl_field, posMessageTable, self.correl_field)
        msgs = 0#keeps track of the number of messages read
        cfRows = mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode)#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        mm.disableTableKeys(self.corpdb, self.dbCursor, posFeatTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting
        for cfRow in cfRows:
            cf_id = cfRow[0]
            mids = set() #currently seen message ids
            freqsPOS = dict() #holds frequency of phrases
            totalTokens = 0

            #grab poses by messages for that correl field:
            for messageRow in self.getMessagesForCorrelField(cf_id, messageTable = posMessageTable, warnMsg = False):
                message_id = messageRow[0]
                pos_message = messageRow[1]
                if not message_id in mids and pos_message:
                    msgs+=1
                    if msgs % dlac.PROGRESS_AFTER_ROWS == 0: #progress update
                        dlac.warn("POS Messages Read: %dk" % int(msgs/1000))
                    mids.add(message_id)

                    #find poses in message
                    if keep_words:
                        # keep the actual word with its POS too
                        pos_list = pos_message.split()
                        pos_list = ['/'.join([w.lower()
                                              for w in i.split('/')[:-1]]+
                                             [i.split('/')[-1]])
                                    for i in pos_list]
                    else:
                        # Just extract the POSes if not needed to keep the ngrams
                        pos_list = [x.split('/')[-1] for x in pos_message.split()]

                    # posDict = find pos frequencies in pos_message
                    for pos in pos_list:
                        # check that we can write pos to table: len(pos) < varchar(*)
                        if len(pos) > min_varchar_length and min_varchar_length != dlac.MAX_SQL_PRINT_CHARS:
                            if len(pos) >= dlac.MAX_SQL_PRINT_CHARS:
                                min_varchar_length = dlac.MAX_SQL_PRINT_CHARS
                            else:
                                min_varchar_length = len(pos)
                            alter_table = True
                        totalTokens += 1
                        if pos in freqsPOS:
                            freqsPOS[pos] += 1
                        else:
                            freqsPOS[pos] = 1

            wsql = """INSERT INTO """+posFeatTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
            totalTokens = float(totalTokens)
            phraseRows = [(k, v, valueFunc((v / totalTokens))) for k, v in freqsPOS.items()] #adds group_norm and applies freq filter
            if alter_table:
                dlac.warn("WARNING: varchar length of feat column is too small, adjusting table.")
                alter_sql = """ALTER TABLE %s CHANGE COLUMN `feat` `feat` VARCHAR(%s)""" %(posFeatTableName, min_varchar_length)
                mm.execute(self.corpdb, self.dbCursor, alter_sql, charset=self.encoding, use_unicode=self.use_unicode)
                alter_table = False
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, phraseRows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

        dlac.warn("Done Reading / Inserting.")

        dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
        mm.enableTableKeys(self.corpdb, self.dbCursor, posFeatTableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys
        dlac.warn("Done\n")
        return posFeatTableName;

    def addOutcomeFeatTable(self, outcomeGetter, tableName = None, valueFunc = lambda d: d):
        """Creates feature table of outcomes

        Parameters
        ----------
        outcomeGetter : OutcomeGetter object
            ?????
        tableName : :obj:`str`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given

        Returns
        -------
        outcomeFeatTableName : str
            Name of created feature table: feat%out_outcomes%corptable$correl_field

        """

        #GET OUTCOMES: (no group freq thresh)
        dlac.warn("GETTING OUTCOMES (Note: No group freq thresh is used) To Insert Into New Feat Table")
        (groups, allOutcomes, controls) = outcomeGetter.getGroupsAndOutcomes(0)
        if controls:
            dlac.warn("controls will be treated just like outcomes (i.e. inserted into feature table)")
            allOutcomes = allOutcomes.intersection(controls)

        #CREATE TABLEs:
        name = '_'.join([k[:1].lower()+k[k.find('_')+1].upper() if '_' in k[:-1] else k[:3] for k in allOutcomes.keys()])
        name = 'out_'+outcomeGetter.outcome_table[:8]+'_'+name[:10]
        outcomeFeatTableName = self.createFeatureTable(name, "VARCHAR(24)", 'DOUBLE', tableName, valueFunc)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        mm.disableTableKeys(self.corpdb, self.dbCursor, outcomeFeatTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting
        for outcome, values in allOutcomes.items():
            dlac.warn("  On %s"%outcome)
            wsql = """INSERT INTO """+outcomeFeatTableName+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""
            phraseRows = [(k, outcome, v, valueFunc(v)) for k, v in values.items()] #adds group_norm and applies freq filter
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, phraseRows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

        dlac.warn("Done Inserting.")

        dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
        mm.enableTableKeys(self.corpdb, self.dbCursor, outcomeFeatTableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys
        dlac.warn("Done\n")
        return outcomeFeatTableName;

    ##TIMEX PROCESSING##

    def addTimexDiffFeatTable(self, dateField=dlac.DEF_DATE_FIELD, tableName = None, serverPort = dlac.DEF_CORENLP_PORT):
        """Creates a feature table of difference between sent-time and time of time expressions, mean, std

        Parameters
        ----------
        dateField : OutcomeGetter object
            ?????
        tableName : :obj:`str`, optional
            ?????
        serverPort : :obj:`str`, optional
            ?????

        Returns
        -------
        featureTableName : str
            Name of created feature table: feat%timex%corptable$correl_field

        """
        ##NOTE: correl_field should have an index for this to be quick

        corenlpServer = jsonrpclib.Server("http://localhost:%d"% serverPort)

        #corenlpServer = getCoreNLPServer(pipeline = ['tokenizer', 'pos',] serverPort = serverPort)

        #CREATE TABLE:
        featureName = 'timex'
        featureTableName = self.createFeatureTable(featureName, "VARCHAR(24)", 'DOUBLE', tableName)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        usql = """SELECT %s FROM %s GROUP BY %s""" % (
            self.correl_field, self.corptable, self.correl_field)
        msgs = 0#keeps track of the number of messages read
        toWrite = [] #group_id, feat, value(and groupnorm)
        cfRows = FeatureExtractor.noneToNull(mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode))#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS: mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting
        written = 0
        for cfRow in cfRows:
            cf_id = cfRow[0]

            mids = set() #currently seen message ids
            timexDiffs = []
            totalWords = 0
            netags = dict()

            #parse message
            for messageRow in self.getMessagesWithFieldForCorrelField(cf_id, dateField, warnMsg = False):
                (message_id, message, messageDT) = messageRow
                if not isinstance(messageDT, datetime.datetime):
                    messageDT = dtParse(messageDT, ignoretz = True)
                #print messageDT #debug
                #print message #debug
                if not message_id in mids and message:
                    msgs+=1
                    if msgs % dlac.PROGRESS_AFTER_ROWS == 0: #progress update
                        dlac.warn("Messages Read: %dk" % int(msgs/1000))
                    message = tc.treatNewlines(message)
                    if self.use_unicode:
                        message = tc.removeNonUTF8(message)
                    else:
                        message = tc.removeNonAscii(message)
                    message = tc.shrinkSpace(message)

                    parseInfo = loads(corenlpServer.parse(message))
                    #print parseInfo #debug
                    newDiffs, thisNEtags, thisWords = self.parseCoreNLPForTimexDiffs(parseInfo, messageDT)
                    #print newDiffs #debug
                    timexDiffs.extend(newDiffs)
                    totalWords += thisWords
                    for tag in thisNEtags:
                        try:
                            netags[tag] += 1
                        except KeyError:
                            netags[tag] = 1

                    mids.add(message_id)

            if netags:
                print(netags, timexDiffs)
                cf_idstr = str(cf_id)
                fTotalMsgs = float(len(mids))
                for tag, value in netags.items():
                    toWrite.append( (cf_idstr, 'ttag:'+tag, value, value/fTotalMsgs) )

                if timexDiffs:
                    #calculate average and std-dev:
                    meanOffset = mean(timexDiffs)
                    absMeanOffset = abs(meanOffset)
                    stdOffset = std(timexDiffs)
                    percentTimex = len(timexDiffs)/fTotalMsgs
                    toWrite.append( (cf_idstr, "meanOffset", meanOffset, meanOffset))
                    toWrite.append( (cf_idstr, "abs_meanOffset", meanOffset, absMeanOffset))
                    if meanOffset > 0:
                        toWrite.append( (cf_idstr, "log_meanOffset", meanOffset, log10(meanOffset+1)))
                        toWrite.append( (cf_idstr, "pos_log_meanOffset", meanOffset, log10(meanOffset+1)))
                    elif meanOffset < 0:
                        toWrite.append( (cf_idstr, "log_meanOffset", meanOffset, -1*log10(absMeanOffset+1)))
                        toWrite.append( (cf_idstr, "neg_log_meanOffset", meanOffset, -1*log10(absMeanOffset+1)))
                    if meanOffset > -0.000694 and meanOffset < 0.000694:
                        toWrite.append( (cf_idstr, "offset_is_zero", 1, 1))
                    toWrite.append( (cf_idstr, "stdOffset", stdOffset, stdOffset))
                    toWrite.append( (cf_idstr, "numTimexes", len(timexDiffs), percentTimex))

            #write if enough
            if len(toWrite) > 1000:
                wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""
                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, toWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                written += len(toWrite)
                toWrite = []
                print("  added %d timex mean or std offsets" % written)

        if len(toWrite) > 0:
            wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, toWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
            written += len(toWrite)
            print("  added %d timex mean or std offsets" % written)
        dlac.warn("Done Reading / Inserting.")

        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys
        dlac.warn("Done\n")
        return featureTableName;

    def addPOSAndTimexDiffFeatTable(self, dateField=dlac.DEF_DATE_FIELD, tableName = None, serverPort = dlac.DEF_CORENLP_PORT, valueFunc = lambda d: int(d)):
        """Creates a feature table of difference between sent-time and time of time expressions, mean, std
        and a POS table version of the table

        Parameters
        ----------
        dateField : OutcomeGetter object
            ?????
        tableName : :obj:`str`, optional
            ?????
        serverPort : :obj:`str`, optional
            ?????
        valueFunc : :obj:`lambda`, optional
            Scales the features by the function given

        Returns
        -------
        featureTableName : str
            Name of created feature table: feat%timex%corptable$correl_field

        """
        ##NOTE: correl_field should have an index for this to be quick
        corenlpServer = jsonrpclib.Server("http://localhost:%d"% serverPort)

        #CREATE TABLES:
        featureName = 'timex'
        featureTableName = self.createFeatureTable(featureName, "VARCHAR(24)", 'DOUBLE', tableName)
        featureName = 'tpos'
        posTableName = self.createFeatureTable(featureName, "VARCHAR(16)", 'INTEGER', tableName)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        usql = """SELECT %s FROM %s GROUP BY %s""" % (
            self.correl_field, self.corptable, self.correl_field)
        msgs = 0#keeps track of the number of messages read
        toWrite = [] #group_id, feat, value(and groupnorm)
        posToWrite = [] #group_id, feat, value(and groupnorm)
        cfRows = FeatureExtractor.noneToNull(mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode))#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS:
            mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#for faster, when enough space for repair by sorting
            mm.disableTableKeys(self.corpdb, self.dbCursor, posTableName, charset=self.encoding, use_unicode=self.use_unicode)
        written = 0
        posWritten = 0
        cfs = 0
        for cfRow in cfRows:
            cfs +=1
            cf_id = cfRow[0]

            mids = set() #currently seen message ids
            timexDiffs = []
            totalWords = 0
            netags = dict()
            postags = Counter()

            #parse message
            for messageRow in self.getMessagesWithFieldForCorrelField(cf_id, dateField, warnMsg = False):
                (message_id, message, messageDT) = messageRow
                if not isinstance(messageDT, datetime.datetime):
                    try:
                        messageDT = dtParse(messageDT, ignoretz = True)
                    except ValueError:
                        dlac.warn("addposandtimexdifftable: skipping message_id %s because date is bad: %s" % (str(message_id), str(messageDT)))
                        continue
                #print messageDT #debug
                #print message #debug
                if not message_id in mids and message:
                    msgs+=1
                    if msgs % dlac.PROGRESS_AFTER_ROWS == 0: #progress update
                        dlac.warn("Messages Read: %dk" % int(msgs/1000))
                    message = tc.treatNewlines(message)
                    if self.use_unicode:
                        message = tc.removeNonUTF8(message)
                    else:
                        message = tc.removeNonAscii(message)
                    message = tc.shrinkSpace(message)

                    parseInfo = loads(corenlpServer.parse(message))

                    #TIMEX PROCESSING
                    newDiffs, thisNEtags, thisWords = self.parseCoreNLPForTimexDiffs(parseInfo, messageDT)
                    #print newDiffs #debug
                    timexDiffs.extend(newDiffs)
                    totalWords += thisWords
                    for tag in thisNEtags:
                        try:
                            netags[tag] += 1
                        except KeyError:
                            netags[tag] = 1

                    #POS PROCESSING:
                    newTags, thisWords = self.parseCoreNLPForPOSTags(parseInfo)
                    postags.update(newTags)

                    mids.add(message_id)

            #add features for this group (cf_id)
            cf_idstr = str(cf_id)
            if netags:
                #print netags, timexDiffs#debug
                fTotalMsgs = float(len(mids))
                for tag, value in netags.items():
                    toWrite.append( (cf_idstr, 'ttag:'+tag, value, value/fTotalMsgs) )

                if timexDiffs:
                    #calculate average and std-dev:
                    meanOffset = mean(timexDiffs)
                    absMeanOffset = abs(meanOffset)
                    stdOffset = std(timexDiffs)
                    percentTimex = len(timexDiffs)/fTotalMsgs
                    toWrite.append( (cf_idstr, "meanOffset", meanOffset, meanOffset))
                    toWrite.append( (cf_idstr, "abs_meanOffset", meanOffset, absMeanOffset))
                    if meanOffset > 0:
                        toWrite.append( (cf_idstr, "log_meanOffset", meanOffset, log10(meanOffset+1)))
                        toWrite.append( (cf_idstr, "pos_log_meanOffset", meanOffset, log10(meanOffset+1)))
                    elif meanOffset < 0:
                        toWrite.append( (cf_idstr, "log_meanOffset", meanOffset, -1*log10(absMeanOffset+1)))
                        toWrite.append( (cf_idstr, "neg_log_meanOffset", meanOffset, -1*log10(absMeanOffset+1)))
                    if meanOffset > -0.000694 and meanOffset < 0.000694:
                        toWrite.append( (cf_idstr, "offset_is_zero", 1, 1))
                    toWrite.append( (cf_idstr, "stdOffset", stdOffset, stdOffset))
                    toWrite.append( (cf_idstr, "numTimexes", len(timexDiffs), percentTimex))

            if postags:
                fTotalWords = float(totalWords)
                for pos, value in postags.items():
                    posToWrite.append( (cf_idstr, pos, value, valueFunc(value/fTotalWords)) )

            #write if enough
            if len(toWrite) > 200:
                wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""
                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, toWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                written += len(toWrite)
                toWrite = []
                print("  TIMEX: added %d records (%d %ss)" % (written, cfs, self.correl_field))

            if len(posToWrite) > 2000:
                wsql = """INSERT INTO """+posTableName+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""
                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, posToWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                posWritten += len(posToWrite)
                posToWrite = []
                print("  TPOS: added %d records (%d %ss)" % (posWritten, cfs, self.correl_field))

        #END CF LOOP
        #WRITE Remaining:
        if len(toWrite) > 0:
            wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, toWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
            written += len(toWrite)
            print("  TIMEX: added %d records (%d %ss)" % (written, cfs, self.correl_field))
        dlac.warn("Done Reading / Inserting.")

        if len(posToWrite) > 0:
            wsql = """INSERT INTO """+posTableName+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, posToWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
            posWritten += len(posToWrite)
            posToWrite = []
            print("  TPOS: added %d records (%d %ss)" % (posWritten, cfs, self.correl_field))

        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys
            mm.enableTableKeys(self.corpdb, self.dbCursor, posTableName, charset=self.encoding, use_unicode=self.use_unicode)#rebuilds keys
        dlac.warn("Done\n")
        return featureTableName;


    @staticmethod
    def parseCoreNLPForPOSTags(parseInfo):
        """returns a dictionary of pos tags and frequencies

        Parameters
        ----------
        parseInfo : dict
            ?????

        Returns
        -------
        posTags, numWords : dict, int
            ?????

        """
        posTags = dict()
        numWords = 0
        try:
            for sent in parseInfo['sentences']:
                for [word, wInfo] in sent['words']:
                    numWords += 1
                    if 'PartOfSpeech' in wInfo:
                        pos = wInfo['PartOfSpeech']
                        try:
                            posTags[pos] += 1
                        except KeyError:
                            posTags[pos] = 1
        except TypeError:
            dlac.warn("CoreNLP: POS: TypeError, Missing sentences or words in %s" % str(parseInfo)[:64])

        return posTags, numWords

    @staticmethod
    def parseCoreNLPForTimexDiffs(parseInfo, messageDT):
        """returns a list differences between datetime and timexes, and normaled ne tags for any timex

        Parameters
        ----------
        parseInfo : dict
            ?????
        messageDT : dict
            ?????

        Returns
        -------
        timexes, netags, numWords : list, set, int
            ?????

        """
        timexes = dict()
        netags = set()
        numWords = 0
        try:
            for sent in parseInfo['sentences']:
                try:
                    for [word, wInfo] in sent['words']:
                        numWords += 1
                        if 'Timex' in wInfo:
                            #get important elements
                            #print wInfo #debug
                            tid, timeexDiff = FeatureExtractor.getTimexDiff(wInfo['Timex'], messageDT)
                            if timeexDiff is not None:
                                timexes[tid] = timeexDiff
                            if 'NormalizedNamedEntityTag' in wInfo:
                            #get important elements
                                netags.add(wInfo['NormalizedNamedEntityTag'])
                except (KeyError, TypeError) as e:
                    dlac.warn("CoreNLP:TimexDiff: KeyError or TypeErrorException: "+ str(e))
                    traceback.print_exception(*sys.exc_info())
                    dlac.warn("sent:" % str(sent)[:48])

        except (KeyError, TypeError):
            dlac.warn("CoreNLP: TimexDiff: Key or Type Error, Missing sentences in %s" % str(parseInfo)[:64])

        return list(timexes.values()), netags, numWords

    @staticmethod
    def getTimexDiff(timexXML, messageDT):
        """?????

        Parameters
        ----------
        timexXML : ?????
            ?????
        messageDT : ?????
            ?????

        Returns
        -------
        tid,  : list, float or None
            ?????

        """
        xmlDoc = xmlParseString(timexXML)
        timeExType = xmlDoc.documentElement.getAttribute('type')
        tid = xmlDoc.documentElement.getAttribute('tid')
        value = xmlDoc.documentElement.getAttribute('value')
        alt_value = xmlDoc.documentElement.getAttribute('alt_value')

        if timeExType.lower() in TimexDateTimeTypes:
            #get timexDT
            timexDT = None
            if value:
                timexDT = FeatureExtractor.timexValueParser(value, messageDT)
            elif alt_value:
                timexDT = FeatureExtractor.timexAltValueParser(alt_value, messageDT)

            #compute difference
            if isinstance(timexDT, datetime.datetime):
                #print "%s - %s" % (str(timexDT), str(messageDT)) #debug
                tDiff = timexDT - messageDT
                return tid, float(float(tDiff.days) + tDiff.seconds/float(86400))

        return tid, None


    @staticmethod
    def timexValueParser(valueStr, messageDT):
        """resolve valueStr to a datetime object

        Parameters
        ----------
        valueStr : ?????
            ?????
        messageDT : ?????
            ?????

        Returns
        -------
        valueStr : str
            ?????

        """
        valueStr = valueStr.replace("XXXX", str(messageDT.year))
        try:
            #print valueStr#DEBUG
            #print messageDT#DEBUG
            return dtParse(valueStr, default=messageDT, ignoretz = True)
        except ValueError:
            value = valueStr
            if value == 'TNI':#night
                return dtParse("T20:00", default=messageDT)
            elif value == 'TEV':#evening
                return dtParse("T19:00", default=messageDT)
            elif value == 'TMO':#morning
                return dtParse("T08:00", default=messageDT)
            elif value == 'TAF':#afternoon
                return dtParse("T15:00", default=messageDT)

            return valueStr

    @staticmethod
    def timexAltValueParser(altValueStr, messageDT):
        """resolve valueStr to a datetime object

        Parameters
        ----------
        altValueStr : ?????
            ?????
        messageDT : ?????
            ?????

        Returns
        -------
        workingDT : ?????
            ?????

        """

        def getTimeDelta(value):
            m = offsetre.match(value) #day month year
            if m:
                num = int(m.group(1))
                unit = m.group(2)
                if unit == 'd':
                    return timedelta(days = num)
                elif unit == 'm':#months
                    return timedelta(days = num*30)
                elif unit == 'y':#years
                    try:
                        return timedelta(days = num*355)
                    except OverflowError:
                        return timedelta(days = 999999998)
                elif unit == 'w':#weeks
                    return timedelta(days = num*7)
            m = toffsetre.match(value) #second, minute hour
            if m:
                num = int(m.group(1))
                unit = m.group(2)
                if unit == 's':
                    return timedelta(seconds = num)
                elif unit == 'm':
                    return timedelta(minutes = num)
                elif unit == 'h':
                    return timedelta(hours = num)
            return None

        commands = altValueStr.split()
        workingDT = messageDT
        i = 0
        while i < len(commands):
            command = commands[i].lower()
            #print " workingDT: %s , Command: %s" % (workingDT, command) #debug
            if command == 'this': #this day
                value = commands[i+1].lower()
                if value == 'ni':#night
                    workingDT = dtParse("T20:00", default=messageDT)
                elif value == 'ev':#evening
                    workingDT = dtParse("T19:00", default=messageDT)
                elif value == 'mo':#morning
                    workingDT = dtParse("T08:00", default=messageDT)
                elif value == 'af':#afternoon
                    workingDT = dtParse("T15:00", default=messageDT)

                i+=1

            elif command == 'intersect':
                value = commands[i+1].lower()
                if value != 'offset':#not an offset (only intersect)
                    if value[0] == 'p':
                        td = getTimeDelta(value)
                        if td:
                            try:
                                workingDT = workingDT + td
                            except OverflowError:#TODO: handle differently
                                pass
                    elif value == 'ni':#night
                        workingDT = dtParse("T20:00", default=workingDT)
                    elif value == 'ev':#evening
                        workingDT = dtParse("T19:00", default=workingDT)
                    elif value == 'mo':#morning
                        workingDT = dtParse("T08:00", default=workingDT)
                    elif value == 'af':#afternoon
                        workingDT = dtParse("T15:00", default=workingDT)

                    else:
                        try:
                            workingDT = dtParse(value, default=workingDT, ignoretz = True)
                        except ValueError:
                            dlac.warn(" timexAltValueParser:bad string for parsing time: %s (from %s)" % (value, altValueStr)) #debug
                    i+=1

            elif command == 'offset' or command == 'next_immediate':
                value = commands[i+1].lower()
                td = getTimeDelta(value)
                if td:
                    try:
                        workingDT = workingDT + td
                    except OverflowError:#TODO: handle differently
                        pass
                i+=1

            elif command[0] != 'p': #datetime
                tempDT = FeatureExtractor.timexValueParser(command, workingDT)
                if isinstance(tempDT, datetime.datetime):
                    workingDT = tempDT

            i+=1
            #print " workingDT: %s " % (workingDT) #debug

        return workingDT


    @staticmethod
    def noneToNull(data):
        """chanes None values to string 'Null'"""
        if data == None:
            return 'Null'
        elif isinstance(data, (list, tuple)):
            newData = []
            for d in data:
                newData.append(FeatureExtractor.noneToNull(d))
            return newData
        else:
            return data

    @staticmethod
    def removeXML(text):
        """Removed XML from text"""
        return dlac.TAG_RE.sub(' ', text)

    @staticmethod
    def removeURL(text):
        """Removes URLs from text"""
        return dlac.URL_RE.sub(' ', text)

    @staticmethod
    def shortenDots(text):
        """Changes None values to string 'Null'"""
        return text.replace('..', '.').replace('..','.')
