import collections
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
import numpy as np
#import imp

#nltk
try:
    from nltk.tree import ParentedTree
    from nltk.corpus import wordnet as wn
except ImportError:
    print("Warning: unable to import nltk.tree or nltk.corpus or nltk.data")

    
#infrastructure
from .dlaWorker import DLAWorker
from . import dlaConstants as dlac
from . import textCleaner as tc
from .mysqlmethods import mysqlMethods as mm

#dataEngine / query
from .database.query import QueryBuilder
from .database.query import Column
from .database.dataEngine import DataEngine

#local / nlp
from .lib.happierfuntokenizing import Tokenizer #Potts tokenizer

try:
    from simplejson import loads
    import jsonrpclib
except ImportError:
    print("Warning: unable to import jsonrpclib or simplejson")
    pass
try:
    from textstat.textstat import textstat
except ImportError:
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
        tokenizer = Tokenizer(preserve_case = not lowercase_only, use_unicode=self.use_unicode)

        #debug:
        #print "valueFunc(30) = %f" % valueFunc(float(30)) #debug

        #CREATE TABLE:
        featureName = str(n)+'gram'
        if not lowercase_only: featureName += 'Up'
        varcharLength = min((dlac.VARCHAR_WORD_LENGTH-(n-1))*n, 255)
        featureTableName = self.createFeatureTable(featureName, "VARCHAR(%d)"%varcharLength, 'INTEGER', tableName, valueFunc, extension = extension)

        if metaFeatures:
            # If metafeats is on, make a metafeature table as well
            mfLength = 16
            mfName = "meta_"+featureName
            mfTableName = self.createFeatureTable(mfName, "VARCHAR(%d)" % mfLength, 'INTEGER', tableName, valueFunc, extension = extension)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        query = self.qb.create_select_query(self.corptable).set_fields([self.correl_field]).group_by([self.correl_field])
        msgs = 0 # keeps track of the number of messages read
        cfRows = FeatureExtractor.noneToNull(query.execute_query())#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows)*n < dlac.MAX_TO_DISABLE_KEYS: self.data_engine.disable_table_keys(featureTableName) #for faster, when enough space for repair by sorting

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
                query = self.qb.create_insert_query(featureTableName).set_values([("group_id",str(cf_id)),("feat",""),("value",""),("group_norm","")])
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
                    #dlac.warn("Inserting rows %d to %d... " % (insert_idx_start, insert_idx_end)) #debug
                    #dlac.warn(insert_rows) #debug
                    query.execute_query(insert_rows)
                    insert_idx_start += dlac.MYSQL_BATCH_INSERT_SIZE
                    insert_idx_end += dlac.MYSQL_BATCH_INSERT_SIZE


                #meta feature table: 
                query = self.qb.create_insert_query(featureTableName).set_values([("group_id",str(cf_id)),("feat",""),("value",""),("group_norm","")])
                totalGrams = float(totalGrams) # to avoid casting each time below
                if self.use_unicode:
                    rows = [(k, v, valueFunc((v / totalGrams))) for k, v in freqs.items() if v >= min_freq] #adds group_norm and applies freq filter
                else:
                    rows = [(k.encode('utf-8'), v, valueFunc((v / totalGrams))) for k, v in freqs.items() if v >= min_freq] #adds group_norm and applies freq filter

                if metaFeatures:
                    mfRows = []
                    mfwsql = self.qb.create_insert_query(mfTableName).set_values([("group_id",str(cf_id)),("feat",""),("value",""),("group_norm","")])
                    avgGramLength = totalChars / totalGrams
                    lenmids=len(mids)
                    avgGramsPerMsg = totalGrams / lenmids
                    mfRows.append( ('_avg'+str(n)+'gramLength', avgGramLength, valueFunc(avgGramLength)) )
                    mfRows.append( ('_avg'+str(n)+'gramsPerMsg', avgGramsPerMsg, valueFunc(avgGramsPerMsg)) )
                    mfRows.append( ('_total'+str(n)+'grams', totalGrams, valueFunc(totalGrams)) )
                    mfRows.append( ('_totalMsgs', lenmids, valueFunc(lenmids)) )
                    mfwsql.execute_query(mfRows)


        dlac.warn("Done Reading / Inserting.")

        if len(cfRows)*n < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            self.data_engine.enable_table_keys(featureTableName)
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
        cfRows = FeatureExtractor.noneToNull(mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file))#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows)*n < dlac.MAX_TO_DISABLE_KEYS: mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting

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
                    mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, insert_rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file);
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
                    mm.executeWriteMany(self.corpdb, self.dbCursor, mfwsql, mfRows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

                # mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding)

        dlac.warn("Done Reading / Inserting.")

        if len(cfRows)*n < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys
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
        cfRows = FeatureExtractor.noneToNull(mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file))#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows)*n < dlac.MAX_TO_DISABLE_KEYS: mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting

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
                    mm.executeWriteMany(self.corpdb, self.dbCursor, mfwsql, mfRows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                #print "\n\n\nROWS TO ADD!!"
                #pprint(rows) #DEBUG
                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, mysql_config_file=self.mysql_config_file)

        dlac.warn("Done Reading / Inserting.")

        if len(cfRows)*n < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys
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
        res = mm.executeGetList(self.corpdb, self.dbCursor, has_column_query, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        has_pmi_column = len(res) > 0
        if has_pmi_column:
            res = mm.executeGetList(self.corpdb, self.dbCursor, "SELECT {} FROM {}.{} WHERE {} < {}".format(colloc_column, colloc_schema, colloc_table, pmi_filter_thresh, pmi_filter_column), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        else:
            dlac.warn("No column named {} found.  Using all collocation in table.".format(pmi_filter_column))
            res = mm.executeGetList(self.corpdb, self.dbCursor, "SELECT {} FROM {}.{}".format(colloc_column, colloc_schema, colloc_table), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
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
        cfRows = FeatureExtractor.noneToNull(mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file))#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS: mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting

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

                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        dlac.warn("Done Reading / Inserting.")

        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys
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

        mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting
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
                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, insert_rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                insert_idx_start += dlac.MYSQL_BATCH_INSERT_SIZE
                insert_idx_end += dlac.MYSQL_BATCH_INSERT_SIZE

        dlac.warn("Done Reading / Inserting.")

        # _warn("This tokenizer took %d seconds"%((datetime.utcnow()-t1).seconds,))

        dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
        mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys
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
        cfRows = mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting
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
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        dlac.warn("Done Reading / Inserting.")

        dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
        mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys
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
        assert mm.tableExists(self.corpdb, self.dbCursor, parseTable, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file), "Need %s table to proceed with phrase featrue extraction " % parseTable
        usql = """SELECT %s FROM %s GROUP BY %s""" % (self.correl_field, parseTable, self.correl_field)
        msgs = 0#keeps track of the number of messages read
        cfRows = mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        mm.disableTableKeys(self.corpdb, self.dbCursor, taggedTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting
        mm.disableTableKeys(self.corpdb, self.dbCursor, phraseTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting
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
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, phraseRows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

            wsql = """INSERT INTO """+taggedTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
            taggedRows = [(k, v, valueFunc((v / totalPhrases))) for k, v in freqsTagged.items()] #adds group_norm and applies freq filter
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, taggedRows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        dlac.warn("Done Reading / Inserting.")

        dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
        mm.enableTableKeys(self.corpdb, self.dbCursor, taggedTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys
        mm.enableTableKeys(self.corpdb, self.dbCursor, phraseTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys
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
        cfRows = FeatureExtractor.noneToNull(mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file))#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS: mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting
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

                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        dlac.warn("Done Reading / Inserting.")

        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys
        dlac.warn("Done\n")
        return featureTableName


    def addEmbTable(self, modelName, tokenizerName, modelClass=None, batchSize=dlac.GPU_BATCH_SIZE, aggregations = ['mean'], layersToKeep = [8,9,10,11], maxTokensPerSeg=255, noContext=True, layerAggregations = ['concatenate'], wordAggregations = ['mean'], keepMsgFeats = False, customTableName = None, valueFunc = lambda d: d):
        '''

        '''
        def addSentTokenized(messageRows):

            try:
                import nltk.data
                import sys
            except ImportError:
                print("warning: unable to import nltk.tree or nltk.corpus or nltk.data")
            sentDetector = nltk.data.load('tokenizers/punkt/english.pickle')
            messages = list(map(lambda x: x[1], messageRows))
            parses = []
            for m_id, message in messageRows:
                parses.append([m_id, json.dumps(sentDetector.tokenize(tc.removeNonUTF8(tc.treatNewlines(message.strip()))))])
            return parses

        dlac.warn("WARNING: new version of BERT and transformer models starts at layer 1 rather than layer 0. Layer 0 is now the input embedding. For example, if you were using layer 10 for the second to last layer of bert-base that is now considered layer 11.")
        ##FIRST MAKE SURE SENTENCE TOKENIZED TABLE EXISTS:
        #sentTable = self.corptable+'_stoks' 
        #assert mm.tableExists(self.corpdb, self.dbCursor, sentTable, charset=self.encoding, use_unicode=self.use_unicode), "Need %s table to proceed with Bert featrue extraction (run --add_sent_tokenized)" % sentTable
        
        sentTok_onthefly = False if self.data_engine.tableExists(self.corptable+'_stoks') else True
        sentTable = self.corptable if sentTok_onthefly else self.corptable+'_stoks'
        if sentTok_onthefly: dlac.warn("WARNING: run --add_sent_tokenized on the message table to avoid tokenizing it every time you generate embeddings")
        

        
        #if len(layerAggregations) > 1:
        #    dlac.warn("AddBert: !!Does not currently support more than one layer aggregation; only using first aggregation!!")
        #    layerAggregations = layerAggregations[:1]

        tokenizerName = modelName if tokenizerName is None else tokenizerName
        
        try: 
            import torch
            from torch.nn.utils.rnn import pad_sequence
            from transformers import AutoConfig, AutoModel, AutoTokenizer
            from transformers import TransfoXLTokenizer, TransfoXLModel, BertModel, BertTokenizer, OpenAIGPTModel, OpenAIGPTTokenizer
            from transformers import GPT2Model, GPT2Tokenizer, XLNetModel, XLNetTokenizer, DistilBertModel, DistilBertTokenizer
            from transformers import RobertaModel, RobertaTokenizer, XLMModel, XLMTokenizer, XLMRobertaModel, XLMRobertaTokenizer
            from transformers import AlbertModel, AlbertTokenizer, T5Model, T5Tokenizer
            #from transformers import ElectraModel, ElectraTokenizer
        except ImportError:
            dlac.warn("warning: unable to import torch or transformers")
            dlac.warn("Please install pytorch and transformers.")
            sys.exit(1)

        SHORTHAND_DICT = { 'bert-base-uncased': 'bert', 'bert-large-uncased': 'bert', 'bert-base-cased': 'bert', 'bert-large-cased': 'bert',
                            'SpanBERT/spanbert-base-cased': 'bert', 'SpanBERT/spanbert-large-cased': 'bert',
                            'allenai/scibert_scivocab_cased': 'bert', 'allenai/scibert_scivocab_uncased': 'bert',# 'transfo-xl-wt103': 'transfoXL',
                            #'openai-gpt': 'OpenAIGPT', 
                            'gpt2': 'GPT2', 'gpt2-medium': 'GPT2', 'gpt2-large': 'GPT2', 'gpt2-xl': 'GPT2',
                            'xlnet-base-cased': 'XLNet', 'xlnet-large-cased': 'XLNet',
                            'roberta-base': 'Roberta', 'roberta-large': 'Roberta', 'roberta-large-mnli': 'Roberta', 
                            'distilroberta-base': 'Roberta', 'roberta-base-openai-detector': 'Roberta', 'roberta-large-openai-detector': 'Roberta',
                            'distilbert-base-uncased': 'DistilBert', 'distilbert-base-cased': 'DistilBert', 
                            'distilbert-base-multilingual-cased': 'DistilBert', 'distilbert-base-cased-distilled-squad': 'DistilBert',
                            'albert-base-v2': 'Albert', 'albert-large-v2': 'Albert', 'albert-xlarge-v2': 'Albert', 'albert-xxlarge-v2': 'Albert',
                            'xlm-roberta-base': 'XLMRoberta', 'xlm-roberta-large': 'XLMRoberta', #'t5-small': 'T5', 't5-base': 'T5', 't5-large': 'T5', 
        }

        MODEL_DICT = {
            #'transfoXL': [TransfoXLModel, TransfoXLTokenizer], #Need to look into tokenization
            'bert' : [BertModel, BertTokenizer],
            'XLNet': [XLNetModel, XLNetTokenizer], 
            #'OpenAIGPT': [OpenAIGPTModel,  OpenAIGPTTokenizer], # Need to fix tokenization [Token type Ids], Doesn't have CLS or SEP
            'Roberta': [RobertaModel, RobertaTokenizer], # Need to fix tokenization [Token type Ids]
            'GPT2': [GPT2Model, GPT2Tokenizer], # Need to fix tokenization [Token type Ids], Doesn't have CLS or SEP
            'DistilBert': [DistilBertModel, DistilBertTokenizer], #Doesn't take Token Type IDS as input
            #'XLM': [XLMModel, XLMTokenizer], #Need to decide on the specific models
            'XLMRoberta': [XLMRobertaModel, XLMRobertaTokenizer],  # Need to fix tokenization [Token type Ids]
            'Albert': [AlbertModel, AlbertTokenizer],
            #'Electra': [ElectraModel, ElectraTokenizer], #Need to understand the discriminator and generator outputs better
            #'T5': [T5Model, T5Tokenizer] #Doesn't take Token Type IDS as input, Doesn't have CLS or SEP, Need to understand how to input the decoder_input_ids/decoder_inputs_embeds
        }

        #print (modelName) #debug
        #Fix AutoModel, runs into a weird positional embedding issue right now.
        #config = AutoConfig.from_pretrained(modelName, output_hidden_states=True)
        #tokenizer = AutoTokenizer.from_pretrained(tokenizerName)
        #model = AutoModel.from_pretrained(modelName, config=config)
        if modelClass is not None: 
            tokenizer = MODEL_DICT[modelClass][1].from_pretrained(tokenizerName)
            model = MODEL_DICT[modelClass][0].from_pretrained(modelName, output_hidden_states=True)
        else:
            tokenizer = MODEL_DICT[SHORTHAND_DICT[tokenizerName]][1].from_pretrained(tokenizerName)
            model = MODEL_DICT[SHORTHAND_DICT[modelName]][0].from_pretrained(modelName, output_hidden_states=True)
        maxTokensPerSeg = tokenizer.max_len_sentences_pair//2
        #Fix for gpt2
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0
        model.eval()
        cuda = True
        try:
            model.to('cuda')
            batch_size=batchSize
        except:
            dlac.warn(" unable to use CUDA (GPU) for BERT")
            batch_size=batchSize
            cuda = False
        dlac.warn("Done.")
        layersToKeep = np.array(layersToKeep, dtype='int')

        #TODO: Change the model name later
        #Need to test noc
        noc = ''
        if noContext: noc = 'noc_'#adds noc to name if no context
        if customTableName is None:
            modelName = modelName.split('/')[-1] if '/' in modelName else modelName
            modelPieces = modelName.split('-')
            modelNameShort = modelPieces[0] + '_' + '_'.join([s[:2] for s in modelPieces[1:]])\
                            + '_' + noc+''.join([str(ag[:2]) for ag in aggregations])+'L'+'L'.join([str(l) for l in layersToKeep])+''.join([str(ag[:2]) for ag in layerAggregations]) + 'n'
        else:
            modelNameShort = customTableName
        if keepMsgFeats:
            embTableName = self.createFeatureTable(modelNameShort, "VARCHAR(12)", 'DOUBLE', None, valueFunc, correlField='message_id')
        else:
            embTableName = self.createFeatureTable(modelNameShort, "VARCHAR(12)", 'DOUBLE', None, valueFunc)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        usql = """SELECT %s FROM %s GROUP BY %s""" % (self.correl_field, sentTable, self.correl_field)
        msgs = 0#keeps track of the number of messages read
        cfRows = FeatureExtractor.noneToNull(self.data_engine.execute_get_list(usql))#SSCursor woudl be better, but it loses connection

        ##iterate through correl_ids (group id):
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        self.data_engine.disable_table_keys(embTableName)#for faster, when enough space for repair by sorting
        lengthWarned = False #whether the length warning has been printed yet
        #Each User: ( #message aggregations, #layers, #Word aggregations, hidden size)
        for cfRow in cfRows:
            #user_id
            cf_id = cfRow[0]
            mids = set() #currently seen message ids
            midList = [] #only for keepMsgFeats

            #grab sents by messages for that correl field:
            messageRows = self.getMessagesForCorrelField(cf_id, messageTable = sentTable, warnMsg=True)
            if sentTok_onthefly:
                messageRows = addSentTokenized(messageRows)
            input_ids = []
            token_type_ids = []
            attention_mask = []
            message_id_seq = [] 
            #stores the sequence of message_id corresponding to the message embeddings for applying aggregation later 
            #along with the sentence 1 and sentence 2 lengths
            for messageRow in messageRows:
                message_id = messageRow[0]
                try:
                    messageSents = loads(messageRow[1])
                except NameError: 
                    dlac.warn("Error: Cannot import jsonrpclib or simplejson in order to get sentences for Bert")
                    sys.exit(1)
                except json.JSONDecodeError:
                    dlac.warn("WARNING: JSONDecodeError on %s. Skipping Message"%str(messageRow))
                    continue
                except:
                    dlac.warn("Warning: cannot load message, skipping")
                    continue

                if ((message_id not in mids) and (len(messageSents) > 0)):
                    msgs+=1
                    subMessages = []
                    if noContext:#break up to run on one word at a time:               
                        for s in messageSents:
                            subMessages.extend([[word] for word in tokenizer.tokenize(s)])
                    else: #keep context; one submessage; subMessages: [[Msg1, Msg2...]]
                        subMessages=[messageSents]

                    for sents in subMessages: #only matters for noContext)
                        #TODO: preprocess to remove newlines
                        sentsTok = [tokenizer.tokenize(s) for s in sents]
                        #print(sentsTok)#debug
                        #check for overlength:
                        i = 0
                        while (i < len(sentsTok)):#while instead of for since array may change size
                            if len(sentsTok[i]) > maxTokensPerSeg: #If the number of tokens is greater than maxTokenPerSeg in a Sentence, split it
                                newSegs = [sentsTok[i][j:j+maxTokensPerSeg] for j in range(0, len(sentsTok[i]), maxTokensPerSeg)]    
                                if not lengthWarned:
                                    dlac.warn("AddEmb: Some segments are too long; splitting up; first example: %s" % str(newSegs))
                                    #lengthWarned = True
                                sentsTok = sentsTok[:i] + newSegs + sentsTok[i+1:]
                                i+=(len(newSegs) - 1)#skip ahead new segments
                            i+=1

                        for i in range(len(sentsTok)):
                            thisPair = sentsTok[i:i+2] #Give two sequences as input
                            try:
                                encoded = tokenizer.encode_plus(thisPair[0], thisPair[1]) if len(thisPair)>1 else tokenizer.encode_plus(thisPair[0])
                            except:
                                dlac.warn("Message pair/ message unreadable. Skipping this....")
                                continue
                                #print(thisPair, message_id)
                                #sys.exit(0)
                                
                            indexedToks = encoded['input_ids']
                            segIds = encoded['token_type_ids'] if 'token_type_ids' in encoded else None

                            input_ids.append(torch.tensor(indexedToks, dtype=torch.long))
                            if 'token_type_ids' in encoded:
                                token_type_ids.append(torch.tensor(segIds, dtype=torch.long))
                            attention_mask.append(torch.tensor([1]*len(indexedToks), dtype=torch.long))

                            if len(thisPair)>1: #Collecting the sentence length of the pair along with their message IDs
                                # If multiple sentences in a message, it will store the message_ids multiple times for aggregating emb later.
                                message_id_seq.append([message_id, len(thisPair[0]), len(thisPair[1])]) 
                            else:
                                message_id_seq.append([message_id, len(thisPair[0]), 0]) 
                
                if msgs % int(dlac.PROGRESS_AFTER_ROWS/5) == 0: #progress update
                    dlac.warn("Messages Read: %.2f k" % (msgs/1000.0))
                mids.add(message_id)
                midList.append(message_id)
            
            #Number of Batches
            num_batches = int(np.ceil(len(input_ids)/batch_size))
            encSelectLayers = []
            #print ('len(input_ids): ',len(input_ids))
            #print ('Num Batches:', num_batches)
            #TODO: Check if len(messageSents) = 0, skip this and print warning
            for i in range(num_batches):
                #Padding for batch input
                input_ids_padded = pad_sequence(input_ids[i*batch_size:(i+1)*batch_size], batch_first = True, padding_value=pad_token_id)
                if len(token_type_ids)>0:
                    token_type_ids_padded = pad_sequence(token_type_ids[i*batch_size:(i+1)*batch_size], batch_first = True, padding_value=0)
                attention_mask_padded = pad_sequence(attention_mask[i*batch_size:(i+1)*batch_size], batch_first = True, padding_value=0)

                if cuda:
                    input_ids_padded = input_ids_padded.to('cuda') 
                    if len(token_type_ids)>0:
                        token_type_ids_padded = token_type_ids_padded.to('cuda') 
                    attention_mask_padded = attention_mask_padded.to('cuda')

                input_ids_padded = input_ids_padded.long()
                if len(token_type_ids)>0:
                    token_type_ids_padded = token_type_ids_padded.long()
                attention_mask_padded = attention_mask_padded.long()
                
                #print (input_ids_padded.shape, token_type_ids_padded.shape, attention_mask_padded.shape)
                encSelectLayers_temp = []
                with torch.no_grad():
                    if len(token_type_ids)>0:
                        encAllLayers = model(input_ids = input_ids_padded, attention_mask = attention_mask_padded,  token_type_ids = token_type_ids_padded)
                    else:
                        encAllLayers = model(input_ids = input_ids_padded, attention_mask = attention_mask_padded)    
                    #Getting all layers output
                    encAllLayers = encAllLayers[-1]            
                    for lyr in layersToKeep: #Shape: (batch_size, max Seq len, hidden dim, #layers)
                        encSelectLayers_temp.append(encAllLayers[int(lyr)].detach().cpu().numpy())

                    #print(encSelectLayers[-1].shape)
                    del encAllLayers, input_ids_padded, attention_mask_padded
                    if len(token_type_ids)>0: del token_type_ids_padded
                    
                encSelectLayers.append(np.transpose(np.array(encSelectLayers_temp),(1,2,3,0)))

            i = 0
            j = 0
            msg_rep = [] #Shape: (num layers, seq Len, hidden_dim)
            while i < len(message_id_seq):
                # Does next embedding also pertain to the same message
                if i == len(message_id_seq)-1:
                    next_msg_same = False
                else:
                    next_msg_same = True if message_id_seq[i][0] == message_id_seq[i+1][0] else False
                if next_msg_same:
                    # Process all sub messages pertaining to a single message at once
                    msg_rep_temp = []
                    # Store the first message's first sentence embedding followed by averaging the second sentence of that message and the first sentence of next message's embedding   
                    msg_rep_temp.append(encSelectLayers[i//batch_size][j%batch_size, :message_id_seq[i][1]])
                    while next_msg_same:
                        enc_batch_number = i//batch_size
                        next_enc_batch_number = (i+1)//batch_size
                        seq_len1 = message_id_seq[i][1]
                        seq_len2 = message_id_seq[i][2]
                        #Shape: (seq2 len, hidden dim, num layers)
                        #Apply mean for the embedding of the sentence that appeared second in the current part and first in the next part
                        sent2enc = (encSelectLayers[enc_batch_number][j%batch_size, seq_len1:seq_len1+seq_len2] + encSelectLayers[next_enc_batch_number][(j+1)%batch_size, :seq_len2])/2
                        #Store the representation
                        msg_rep_temp.append(sent2enc)
                        #print (message_id_seq[i], sent2enc.shape) #debug
                        i+=1
                        j+=1
                        #Check if the next part of the message has single sentence, if yes, break 
                        if message_id_seq[i][2] == 0:
                            i+=1
                            j+=1
                            break                        
                        next_msg_same = True if message_id_seq[i][0] == message_id_seq[i+1][0] else False
                    #Store all the representation as a list
                    msg_rep.append(msg_rep_temp)
                else:
                    # Single message representations.
                    enc_batch_number = i//batch_size
                    seq_len = message_id_seq[i][1]
                    #Store the message representation
                    msg_rep_temp = encSelectLayers[enc_batch_number][j%batch_size, :seq_len] #Shape: (seq len, hidden dim, #layers)
                    #print (seq_len, msg_rep_temp.shape) #debug
                    i+=1
                    j+=1
                    msg_rep.append([msg_rep_temp])

            #Layer aggregation followed by word aggregation
            user_rep = [] #(num msgs, hidden_dim, lagg)
            for i in range(len(msg_rep)):#Iterating through messages
                sent_rep = []
                for j in range(len(msg_rep[i])): #Iterate through the submessages to apply layer aggregation. 
                    sub_msg = msg_rep[i][j]
                    sub_msg_lagg = []
                    for lagg in layerAggregations:
                        if lagg == 'concatenate':
                            sub_msg_lagg.append(sub_msg) #(seq len, hidden dim, num layers)
                        else:
                            sub_msg_lagg.append(eval("np."+lagg+"(sub_msg, axis=-1)").reshape(sub_msg.shape[0], sub_msg.shape[1], 1) )#(seq len, hidden dim, 1)
                        #Shape: (seq len, hidden dim, (num_layers*(concatenate==True)+(sum(other layer aggregations))))
                        #Example: lagg = [mean, min, concatenate], layers = [8,9]; Shape: (seq len, hidden dim, 2 + 1 + 1)
                        sub_msg_lagg_ = np.concatenate(sub_msg_lagg, axis=-1) 
                    #Getting the mean of all tokens representation
                    #TODO: add word agg list and do eval
                    sub_msg_lagg_wagg = np.mean(sub_msg_lagg_, axis=0) #Shape: (hidden dim, lagg)
                    #ReShaping: (1, hidden dim, lagg)
                    sub_msg_lagg_wagg = sub_msg_lagg_wagg.reshape(1, sub_msg_lagg_wagg.shape[0], sub_msg_lagg_wagg.shape[1]) 
                    #Sentence representations
                    sent_rep.append(sub_msg_lagg_wagg)
                #Accumulate all the sentence representation of a user
                user_rep.append(np.mean(np.concatenate(sent_rep, axis=0), axis=0)) 

            user_rep = np.array(user_rep)
            if user_rep.shape[0] == 0:
                continue
            #Flatten the features [layer aggregations] to a single dimension.
            user_rep = user_rep.reshape(user_rep.shape[0], -1)
            if len(user_rep)>0:
                embFeats = dict()

                if keepMsgFeats: #just store message embeddings
                    embRows = []
                    for mid, msg in zip(midList, user_rep):
                        embRows.extend([(str(mid), str(k), v, valueFunc(v)) for (k, v) in enumerate(msg)])
                    wsql = """INSERT INTO """+embTableName+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""
                    mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, embRows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                    
                else:#Applying message aggregations
                    for ag in aggregations:
                        thisAg = eval("np."+ag+"(user_rep, axis=0)")
                        embFeats.update([(str(k)+ag[:2], v) for (k, v) in enumerate(thisAg)])
                
                        #wsql = """INSERT INTO """+embTableName+""" (group_id, feat, value, group_norm) values ('"""+str(cf_id)+"""', %s, %s, %s)"""
                        insert_idx_start = 0
                        insert_idx_end = dlac.MYSQL_BATCH_INSERT_SIZE
                        query = self.qb.create_insert_query(embTableName).set_values([("group_id",str(cf_id)),("feat",""),("value",""),("group_norm","")])
                        embRows = [(k, float(v), valueFunc(float(v))) for k, v in embFeats.items()] #adds group_norm and applies freq filter
                        while insert_idx_start < len(embRows):
                            insert_rows = embRows[insert_idx_start:min(insert_idx_end, len(embRows))]
                            query.execute_query(insert_rows)
                            insert_idx_start += dlac.MYSQL_BATCH_INSERT_SIZE
                            insert_idx_end += dlac.MYSQL_BATCH_INSERT_SIZE
                        #self.data_engine.execute_write_many(wsql, embRows)
            
        dlac.warn("Done Reading / Inserting.")
        dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
        self.data_engine.enable_table_keys(embTableName)#rebuilds keys
        dlac.warn("Done\n")
        return embTableName;       

    
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
        
        try:
            fk_score = textstat.flesch_kincaid_grade
        except NameError: 
            dlac.warn("Cannot import textstat (cannot use addFleschKincaidTable)")
            sys.exit(1)

        #CREATE TABLE:
        featureName = 'flkin'
        featureTableName = self.createFeatureTable(featureName, "VARCHAR(16)", 'FLOAT', tableName, valueFunc)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        usql = """SELECT %s FROM %s GROUP BY %s""" % (
            self.correl_field, self.corptable, self.correl_field)
        msgs = 0 # keeps track of the number of messages read
        cfRows = FeatureExtractor.noneToNull(mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file))#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS: mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting

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
                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        dlac.warn("Done Reading / Inserting.")

        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys
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
            Sting appended to end of table name

        """
        #create table name
        if not tableName:
            tableName = 'feat$'+featureName+'$'+self.corptable+'$'+self.correl_field
            if 'cat_' in featureName:
                if valueFunc and round(valueFunc(16)) != 16:
                    tableName += '$' + str(16)+'to'+"%d"%round(valueFunc(16))
                wt_abbrv = self.wordTable.split('$')[1][:4]
                tableName += '$' + wt_abbrv
                try:#make sure it can support "_intercept"
                    if int(re.findall(r'\((\d+)\)', featureType)[0]) < 10:
                        #print(featureType)#debug
                        featureType = re.sub(r'\(\d+\)', '(10)', featureType)
                        #print(featureType)#debug
                except IndexError:
                    warn("feature extractor: unable to check if category name can support _intercept")

            else:
                if valueFunc and round(valueFunc(16)) != 16:
                    tableName += '$' + str(16)+'to'+"%d"%round(valueFunc(16))
            if extension:
                tableName += '$' + extension

        #find correl_field type:
        where_conditions = """table_schema='%s' AND table_name='%s' AND column_name='%s'"""%(self.corpdb, self.corptable, self.correl_field)
        query = self.qb.create_select_query("information_schema.columns").set_fields(["column_type"]).where(where_conditions)
        try:
            correlField = self.getCorrelFieldType(self.correl_field) if not correlField else correlField
            correl_fieldType = query.execute_query()[0][0] if not correlField else correlField
        except IndexError:
            dlac.warn("Your message table '%s' (or the group field, '%s') probably doesn't exist (or the group field)!" %(self.corptable, self.correl_field))
            raise IndexError("Your message table '%s' probably doesn't exist!" % self.corptable)

        featureTypeAndEncoding = featureType
        if featureType[0].lower() == 't' or featureType[0].lower() == 'v':
            #string type; add unicode: 
            featureTypeAndEncoding = "%s CHARACTER SET %s COLLATE %s" % (featureType, self.encoding, dlac.DEF_COLLATIONS[self.encoding.lower()])            
            if self.db_type == "sqlite":
                featureTypeAndEncoding = featureType 
            
        #create sql
        dropTable = self.qb.create_drop_query(tableName)
        createTable = self.qb.create_createTable_query(tableName).add_columns([Column("id","BIGINT(16)", unsigned=True, primary_key=True, nullable=False, auto_increment=True),Column("group_id", correl_fieldType), Column("feat", featureTypeAndEncoding), Column("value", valueType), Column("group_norm", "DOUBLE")]).add_mul_keys([("correl_field", "group_id"), ("feature", "feat")]).set_character_set(self.encoding).set_collation(dlac.DEF_COLLATIONS[self.encoding.lower()]).set_engine(dlac.DEF_MYSQL_ENGINE)

        #run sql
        dropTable.execute_query()
        createTable.execute_query()

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
        mm.execute(self.corpdb, self.dbCursor, drop, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

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
        rows = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
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
            attributeRows = mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)[0]
            if attributeRows[0]:
                rows = [(feat, topic, str(feat_cat_weight[feat][topic]*attributeRows[1])) for topic in feat_cat_weight[feat]]

            rowsToInsert.extend(rows)

            if len(rowsToInsert) > dlac.MYSQL_BATCH_INSERT_SIZE:
                mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                rowsToInsert = []
            featIdCounter += 1
            if featIdCounter % reporting_int == 0:
                dlac.warn("%d out of %d features processed; %2.2f complete"%(featIdCounter, len(feat_cat_weight), float(featIdCounter)/len(feat_cat_weight)))

        if len(rowsToInsert) > 0:
            mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            rowsToInsert = []
            dlac.warn("%d out of %d features processed; %2.2f complete"%(featIdCounter, len(feat_cat_weight), float(featIdCounter)/len(feat_cat_weight)))

#        mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

        return tableName

    def addLexiconFeat(self, lexiconTableName, lowercase_only=dlac.LOWERCASE_ONLY, tableName=None,
                       valueFunc=lambda x: float(x), isWeighted=False, featValueFunc=lambda d: float(d),
                       extension=None, lexicon_weighting=False):
        """Creates a feature table given a 1gram feature table name, a lexicon table / database name

        Parameters
        ----------
        lexiconTableName : str
            Name of base lexicon table
        lowercase_only : boolean
            use only lowercase charngrams if True
        tableName : :obj:`str`, optional
            Prespecified name of extracted lexicon feature table, use at own risk
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
        if self.db_type == "sqlite":
            self.data_engine.execute("attach '%s.db' as lexiconDB"%(self.lexicondb))
            sql = "SELECT * FROM lexiconDB.%s"%(lexiconTableName)
        else:
            sql = "SELECT * FROM %s.%s"%(self.lexicondb, lexiconTableName)
        rows = self.data_engine.execute_get_list(sql)
        #rows = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        categories = set()
        lexiconHasWildCard = False
        warnedAboutWeights = False
        max_category_string_length = len("_intercept") # previously -1
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
        lexiconTableNameToCreate = lexiconTableName
        if isWeighted:
            lexiconTableNameToCreate += "_w"
        if featValueFunc(16) != 16:
            lexiconTableNameToCreate += "_16to"+str(int(featValueFunc(16)))
        if lexicon_weighting:
            lexiconTableNameToCreate += "_lw"


        tableName = self.createFeatureTable("cat_%s" % lexiconTableNameToCreate, 'VARCHAR(%d)' %
                                            max_category_string_length, 'INTEGER', tableName, valueFunc,
                                            extension=extension)


        #4. grab all distinct group ids
        wordTable = self.getWordTable()
        dlac.warn("WORD TABLE %s"%(wordTable,))

        assert self.data_engine.tableExists(wordTable), "Need to create word table to extract the lexicon: %s" % wordTable
        #assert mm.tableExists(self.corpdb, self.dbCursor, wordTable, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file), "Need to create word table to extract the lexicon: %s" % wordTable
        sql = "SELECT DISTINCT group_id FROM %s" % wordTable
        groupIdRows = self.data_engine.execute_get_list(sql)
        #groupIdRows = mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        #5. disable keys on that table if we have too many entries
        #if (len(categories)* len(groupIdRows)) < dlac.MAX_TO_DISABLE_KEYS:
        self.data_engine.disable_table_keys(tableName)
        #mm.disableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file) #for faster, when enough space for repair by sorting

        #6. iterate through source feature table by group_id (fixed, column name will always be group_id)
        rowsToInsert = []

        isql = self.qb.create_insert_query(tableName).set_values([("group_id",""),("feat",""),("value",""),("group_norm","")])
        #isql = "INSERT IGNORE INTO "+tableName+" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"

        reporting_percent = 0.01
        reporting_int = max(floor(reporting_percent * len(groupIdRows)), 1)
        groupIdCounter = 0
        for groupIdRow in groupIdRows:

            groupId = groupIdRow[0]

            #i. create the group_id category counts & keep track of how many features they have total
            cat_to_summed_value = dict()
            cat_to_function_summed_weight = dict()
            cat_to_function_summed_weight_gn = {}
            if isinstance(groupId, str):
                sql = "SELECT group_id, feat, value, group_norm FROM %s WHERE group_id LIKE '%s'"%(wordTable, groupId)
            else:
                sql = "SELECT group_id, feat, value, group_norm FROM %s WHERE group_id = %d"%(wordTable, groupId)

            try:
                attributeRows = self.data_engine.execute_get_list(sql)
                #attributeRows = mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            except:
                print(groupId)
                sys.exit()

            totalFeatCountForThisGroupId = 0

            totalFunctionSumForThisGroupId = float(0.0)
            totalWordsInLexForThisGroupId = float(0.0)

            if lexicon_weighting:
                totals = collections.defaultdict(lambda: collections.defaultdict(float))
                for gid, feat, value, _ in attributeRows:
                    if feat in feat_cat_weight:
                        for category in feat_cat_weight[feat]:
                            totals[gid][category] += value
                    if lexiconHasWildCard:  # check wildcard matches
                        for endI in range(3, len(feat) + 1):
                            featWild = feat[0:endI] + '*'
                            if featWild in feat_cat_weight:
                                for category in feat_cat_weight[featWild]:
                                    totals[gid][category] += value

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
                        group_norm = value / float(totals[gid][category])
                    except NameError:  # not using lexicon_weighting
                        pass
                    except ZeroDivisionError:  # should be impossible; a cat w/o a total shouldn't have this feat in it
                        dlac.warn('Something is wrong: feature {feat} appears in empty category {cat} for '
                                  'group_id {gid}'.format(feat=feat, cat=category, gid=gid))
                        group_norm = 0.0

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

            
            if _intercepts:
                for k, v in _intercepts.items():
                    try:
                        cat_to_function_summed_weight_gn[k] += v
                    except KeyError:
                        cat_to_summed_value[k] = 0
                        cat_to_function_summed_weight_gn[k] = v

            # Applying the featValueFunction to the group_norm,
            if self.use_unicode:
                rows = [(gid, k, cat_to_summed_value[k], valueFunc(v)) for k, v in cat_to_function_summed_weight_gn.items()]
            else:
                rows = [(gid, k.encode('utf-8'), cat_to_summed_value[k], valueFunc(v)) for k, v in cat_to_function_summed_weight_gn.items()]
            
            # if lex has *no* intercept, add '_intercept' for each group_id
            if not _intercepts: rows.append((gid, '_intercept', 1, 1.0))

            # iii. Insert data into new feautre table
            # Add new data to rows to be inserted into the database
            # Check if size is big enough for a batch insertion (10,000?), if so insert and clear list
            rowsToInsert.extend(rows)
            if len(rowsToInsert) > dlac.MYSQL_BATCH_INSERT_SIZE:
                isql.execute_query(rowsToInsert)
                #self.data_engine.execute_write_many(isql, rowsToInsert)
                #mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                rowsToInsert = []
            groupIdCounter += 1
            if groupIdCounter % reporting_int == 0:
                dlac.warn("%d out of %d group Id's processed; %2.2f complete"%(groupIdCounter, len(groupIdRows), float(groupIdCounter)/len(groupIdRows)))

        #7. if any data in the data_to_insert rows, insert the data and clear the list
        if len(rowsToInsert) > 0:
            isql.execute_query(rowsToInsert)
            #mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            rowsToInsert = []
            dlac.warn("%d out of %d group Id's processed; %2.2f complete"%(groupIdCounter, len(groupIdRows), float(groupIdCounter)/len(groupIdRows)))

        #8. enable keys on the new feature table
        #if (len(categories)* len(groupIdRows)) < dlac.MAX_TO_DISABLE_KEYS:
        self.data_engine.enable_table_keys(tableName)
        #mm.enableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys

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
        assert mm.tableExists(self.corpdb, self.dbCursor, wordTable, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file), "Need to create word table to apply groupThresh: %s" % wordTable
        sql = "SELECT DISTINCT group_id FROM %s"%wordTable
        groupIdRows = mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        #4. disable keys on that table if we have too many entries
        mm.disableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting

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
            attributeRows = mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

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
                mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                rowsToInsert = []
            groupIdCounter += 1
            if groupIdCounter % reporting_int == 0:
                dlac.warn("%d out of %d group Id's processed; %2.2f complete"%(groupIdCounter, len(groupIdRows), float(groupIdCounter)/len(groupIdRows)))

        #6. if any data in the data_to_insert rows, insert the data and clear the list
        if len(rowsToInsert) > 0:
            mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            rowsToInsert = []

        #7. enable keys on the new feature table
        mm.enableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys

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
        assert mm.tableExists(self.corpdb, self.dbCursor, wordTable, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file), "Need to create 1gram with default scaling (16to16) table to apply groupThresh: %s" % wordTable

        #3.2 check that the POS table exists
        if not pos_table:
            pos_table = "feat$1gram_pos$%s$%s" %(self.corptable, self.correl_field)
            if not mm.tableExists(self.corpdb, self.dbCursor, wordTable, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file):
                pos_table = "feat$1gram_pos$%s$%s$16to16" %(self.corptable, self.correl_field)
        dlac.warn("POS TABLE: %s"%(pos_table,))
        assert mm.tableExists(self.corpdb, self.dbCursor, pos_table, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file), "Need to create POS table to apply functionality: %s" % pos_table
        sql = "SELECT DISTINCT group_id FROM %s"%pos_table
        groupIdRows = mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        #4. disable keys on that table if we have too many entries
        mm.disableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting

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
            attributeRows = mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

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
                mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                rowsToInsert = []
            groupIdCounter += 1
            if groupIdCounter % reporting_int == 0:
                dlac.warn("%d out of %d group Id's processed; %2.2f complete"%(groupIdCounter, len(groupIdRows), float(groupIdCounter)/len(groupIdRows)))

        #6. if any data in the data_to_insert rows, insert the data and clear the list
        if len(rowsToInsert) > 0:
            mm.executeWriteMany(self.corpdb, self.dbCursor, isql, rowsToInsert, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            rowsToInsert = []

        #7. enable keys on the new feature table
        mm.enableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys

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
        posMessageTable = self.corptable
        if posMessageTable[-4:] != '_pos':
            posMessageTable = self.corptable+'_pos'
        assert mm.tableExists(self.corpdb, self.dbCursor, posMessageTable, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file), "Need %s table to proceed with pos featrue extraction " % posMessageTable
        usql = """SELECT %s FROM %s GROUP BY %s""" % (self.correl_field, posMessageTable, self.correl_field)
        msgs = 0#keeps track of the number of messages read
        cfRows = mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        mm.disableTableKeys(self.corpdb, self.dbCursor, posFeatTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting
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


                    pos_list = []
                    if posMessageTable[-4:] != 'tpos':
                        if keep_words:
                            dlac.warn("keep words not implemented yet for tweetpos tags")
                        else:##TODO: Debug; make sure this works
                            try:
                                pos_list = loads(pos_message)['tags'] 
                            except:
                                pos_list = []
                    else:
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
                mm.execute(self.corpdb, self.dbCursor, alter_sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                alter_table = False
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, phraseRows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        dlac.warn("Done Reading / Inserting.")

        dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
        mm.enableTableKeys(self.corpdb, self.dbCursor, posFeatTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys
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
        mm.disableTableKeys(self.corpdb, self.dbCursor, outcomeFeatTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting
        for outcome, values in allOutcomes.items():
            dlac.warn("  On %s"%outcome)
            wsql = """INSERT INTO """+outcomeFeatTableName+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""
            phraseRows = [(k, outcome, v, valueFunc(v)) for k, v in values.items()] #adds group_norm and applies freq filter
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, phraseRows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        dlac.warn("Done Inserting.")

        dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
        mm.enableTableKeys(self.corpdb, self.dbCursor, outcomeFeatTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys
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

        try:
            corenlpServer = jsonrpclib.Server("http://localhost:%d"% serverPort)
        except NameError: 
            dlac.warn("Cannot import jsonrpclib or simplejson")
            sys.exit(1)

        #corenlpServer = getCoreNLPServer(pipeline = ['tokenizer', 'pos',] serverPort = serverPort)

        #CREATE TABLE:
        featureName = 'timex'
        featureTableName = self.createFeatureTable(featureName, "VARCHAR(24)", 'DOUBLE', tableName)

        #SELECT / LOOP ON CORREL FIELD FIRST:
        usql = """SELECT %s FROM %s GROUP BY %s""" % (
            self.correl_field, self.corptable, self.correl_field)
        msgs = 0#keeps track of the number of messages read
        toWrite = [] #group_id, feat, value(and groupnorm)
        cfRows = FeatureExtractor.noneToNull(mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file))#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS: mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting
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

                    try:
                        parseInfo = loads(corenlpServer.parse(message))
                    except NameError: 
                        dlac.warn("Cannot import jsonrpclib or simplejson")
                        sys.exit(1)
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
                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, toWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                written += len(toWrite)
                toWrite = []
                print("  added %d timex mean or std offsets" % written)

        if len(toWrite) > 0:
            wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, toWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            written += len(toWrite)
            print("  added %d timex mean or std offsets" % written)
        dlac.warn("Done Reading / Inserting.")

        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys
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
        try:
            corenlpServer = jsonrpclib.Server("http://localhost:%d"% serverPort)
        except NameError: 
            dlac.warn("Cannot import jsonrpclib or simplejson")
            sys.exit(1)

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
        cfRows = FeatureExtractor.noneToNull(mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file))#SSCursor woudl be better, but it loses connection
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS:
            mm.disableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#for faster, when enough space for repair by sorting
            mm.disableTableKeys(self.corpdb, self.dbCursor, posTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
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

                    try:
                        parseInfo = loads(corenlpServer.parse(message))
                    except NameError: 
                        dlac.warn("Cannot import jsonrpclib or simplejson")
                        sys.exit(1)
                    except ConnectionRefusedError as cre:
                        dlac.warn("Add Timex POS: Can not connect to timex parser server on port: %d\n"%serverPort+str(cre))
                        sys.exit(1)

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
                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, toWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                written += len(toWrite)
                toWrite = []
                print("  TIMEX: added %d records (%d %ss)" % (written, cfs, self.correl_field))

            if len(posToWrite) > 2000:
                wsql = """INSERT INTO """+posTableName+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""
                mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, posToWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                posWritten += len(posToWrite)
                posToWrite = []
                print("  TPOS: added %d records (%d %ss)" % (posWritten, cfs, self.correl_field))

        #END CF LOOP
        #WRITE Remaining:
        if len(toWrite) > 0:
            wsql = """INSERT INTO """+featureTableName+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, toWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            written += len(toWrite)
            print("  TIMEX: added %d records (%d %ss)" % (written, cfs, self.correl_field))
        dlac.warn("Done Reading / Inserting.")

        if len(posToWrite) > 0:
            wsql = """INSERT INTO """+posTableName+""" (group_id, feat, value, group_norm) values (%s, %s, %s, %s)"""
            mm.executeWriteMany(self.corpdb, self.dbCursor, wsql, posToWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            posWritten += len(posToWrite)
            posToWrite = []
            print("  TPOS: added %d records (%d %ss)" % (posWritten, cfs, self.correl_field))

        if len(cfRows) < dlac.MAX_TO_DISABLE_KEYS:
            dlac.warn("Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).")
            mm.enableTableKeys(self.corpdb, self.dbCursor, featureTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys
            mm.enableTableKeys(self.corpdb, self.dbCursor, posTableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#rebuilds keys
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
