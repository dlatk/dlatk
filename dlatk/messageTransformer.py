import ast
import csv
import json
import time
import multiprocessing
import imp
import sys
import os
import re

from io import StringIO

#infrastructure
from .dlaWorker import DLAWorker
from . import dlaConstants as dlac
from . import textCleaner as tc
from .mysqlmethods import mysqlMethods as mm
from .lib.happierfuntokenizing import Tokenizer #Potts tokenizer
try:
    from .lib.StanfordParser import StanfordParser
except ImportError:
    dlac.warn("Cannot import StanfordParser (interface with the Stanford Parser)")
    pass

#nltk
try:
    import nltk.data
except ImportError:
    print("warning: unable to import nltk.tree or nltk.corpus or nltk.data")
try:
    from .lib.TweetNLP import TweetNLP
except ImportError:
    dlac.warn("Cannot import TweetNLP (interface with CMU Twitter tokenizer / pos tagger)")
    pass


class MessageTransformer(DLAWorker):
    """Deals with message tables .....

    Returns
    -------
    MessageTransformer object

    Examples
    --------

    """

    groupsAtTime = 100

    def _createTable(self, tableName, modify='', modify_id=''):
        drop = """DROP TABLE IF EXISTS %s""" % (tableName)
        mm.execute(self.corpdb, self.dbCursor, drop, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        sql = """CREATE TABLE %s like %s""" % (tableName, self.corptable)
        mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        if modify:
            alter = """ALTER TABLE %s MODIFY %s %s""" % (tableName, self.message_field, modify)
            mm.execute(self.corpdb, self.dbCursor, alter, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        if modify_id:#modify message id field:
            alter = """ALTER TABLE %s MODIFY %s %s""" % (tableName, self.messageid_field, modify_id)
            mm.execute(self.corpdb, self.dbCursor, alter, charset=self.encoding, use_unicode=self.use_unicode)

        mm.standardizeTable(self.corpdb, self.dbCursor, tableName, collate=dlac.DEF_COLLATIONS[self.encoding.lower()], engine=dlac.DEF_MYSQL_ENGINE, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        mm.disableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        columnNames = list(mm.getTableColumnNameTypes(self.corpdb, self.dbCursor, self.corptable, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file).keys())
        messageIndex = columnNames.index(self.message_field)
        messageIdIndex = columnNames.index(self.messageid_field)
        return columnNames, messageIndex, messageIdIndex

    def _findAllGroups(self):
        usql = """SELECT %s FROM %s GROUP BY %s""" % (self.correl_field, self.corptable, self.correl_field)
        cfRows = [r[0] for r in mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)]
        dlac.warn("finding messages for %d '%s's"%(len(cfRows), self.correl_field))
        return cfRows

    def _getMsgsForGroups(self, groups, columnNames, messageIndex):
        sql = """SELECT %s from %s where %s IN ('%s')""" % (','.join(columnNames), self.corptable, self.correl_field, "','".join(str(g) for g in groups))
        rows = list(mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file))#, False)
        return rows

    def _writeMsgsForGroups(self, rows, parses, messageIndex, tableName, columnNames):
        insert_idx_start = 0
        insert_idx_end = dlac.MYSQL_BATCH_INSERT_SIZE

        #add msgs into new tables
        sql = """INSERT INTO """+tableName+""" ("""+', '.join(columnNames)+\
                """) VALUES ("""  +", ".join(['%s']*len(columnNames)) + """)"""
        for i in range(len(rows)):
            rows[i] = list(rows[i])
            rows[i][messageIndex] = str(parses[i])

        while insert_idx_start < len(rows):
            dataToWrite = rows[insert_idx_start:min(insert_idx_end, len(rows))]
            mm.executeWriteMany(self.corpdb, self.dbCursor, sql, dataToWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            insert_idx_start += dlac.MYSQL_BATCH_INSERT_SIZE
            insert_idx_end += dlac.MYSQL_BATCH_INSERT_SIZE

    def _addTransformedMessages(self, tableName, transformation_func):
        
        #Create Table:
        columnNames, messageIndex, messageIdIndex = self._createTable(tableName, modify='LONGTEXT')

        #find all groups
        cfRows = self._findAllGroups()

        #iterate through groups in chunks
        groupsWritten = 0
        for groups in dlac.chunks(cfRows, self.groupsAtTime):

            #get msgs for groups:
            rows = self._getMsgsForGroups(groups, columnNames, messageIndex)
            messages = [r[messageIndex] for r in rows]

            if messages:
                #tokenize msgs:
                parses = [json.dumps(transformation_func(m)) for m in messages]
                self._writeMsgsForGroups(rows, parses, messageIndex, tableName, columnNames)

                groupsWritten += self.groupsAtTime
                if groupsWritten % 100 == 0:
                    dlac.warn("  %.1fk %ss' messages tagged and written" % (groupsWritten/float(1000), self.correl_field))
            else:
                dlac.warn("   Warning: No messages for:" + str(groups))

        #re-enable keys:
        mm.enableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        return tableName

    def insertLDARows(self, ldas, tableName, columnNames, messageIndex, messageIdIndex):
        """?????

        Parameters
        ----------
        ldas : ?????
            ?????
        tableName : ?????
            ?????
        columnNames : ?????
            ?????
        messageIndex : ?????
            ?????
        messageIdIndex : ?????
            ?????

        """
        message_ids = [str(msg_id) for msg_id in ldas.keys()]
        sql = """SELECT %s from %s where %s IN ('%s')""" % (
            ','.join(columnNames), self.corptable, self.messageid_field,
            "','".join(message_ids))

        rows = list(mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file))

        #generate row data:
        newRows = []
        for row in rows:
            if row[messageIndex] and not row[messageIndex].isspace():
                newRow = list(row)
                newRow[messageIndex] = json.dumps(ldas[str(row[messageIdIndex])])
                newRows.append(newRow)
                
        #insert
        sql = """INSERT INTO """+tableName+""" ("""+', '.join(columnNames)+\
            """) VALUES ("""  +", ".join(['%s']*len(columnNames)) + """)"""
        mm.executeWriteMany(self.corpdb, self.dbCursor, sql, newRows, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)


    def addLDAMessages(self, ldaStatesFile, ldaStatesName=None):
        """Creates a LDA topic version of message table

        Parameters
        ----------
        ldaStatesFile : str
            Path to file created with addMessageID.py

        Returns
        -------
        tableName : str
            Name of LDA message table: corptable_lda$ldaStatesFileBaseName
        """
        print('LDA states file: {}'.format(ldaStatesFile))
        fin = open(ldaStatesFile, 'r') #done first so throws error if not existing
        if ldaStatesName is None:
            ldaStatesName = os.path.splitext(os.path.basename(ldaStatesFile))[0].replace('-', '_')
        tableName = "%s_lda$%s" %(self.corptable, ldaStatesName)

        #Create Table:
        columnNames, messageIndex, messageIdIndex = self._createTable(tableName, modify='LONGTEXT')

        commentLine = re.compile('^\#')
        ldaColumnLabels = ['doc', 'message_id', 'index', 'term_id', 'term', 'topic_id']
        labelRange = list(range(len(ldaColumnLabels)))
        ldas = dict() #stored ldas currently being looked at
        msgsAtTime = 100
        msgsWritten = 0

        ##iterate through file:
        for line in fin:
            if not commentLine.match(line):
                line = line.strip()
                fields = line.split()
                currentLDA = dict( [(ldaColumnLabels[i], fields[i]) for i in  labelRange])
                currentId = currentLDA['message_id']
                if currentId not in ldas:
                    if  len(ldas) >= msgsAtTime:#insert
                        self.insertLDARows(ldas, tableName, columnNames, messageIndex, messageIdIndex)
                        ldas = dict() #reset memory
                        msgsWritten += msgsAtTime
                        if msgsWritten % 20000 == 0:
                            dlac.warn("  %.1fk messages' lda topics written" % (msgsWritten/float(1000)))
                    ldas[currentId] = [currentLDA]
                else:
                    ldas[currentId].append(currentLDA)

        #write remainder:
        self.insertLDARows(ldas, tableName, columnNames, messageIndex, messageIdIndex)

        #re-enable keys:
        mm.enableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        return tableName

    def addTokenizedMessages(self):
        """Creates a parsed version of the message table

        Returns
        -------
        tableName : str
            Name of tokenized message table: corptable_tok.
        """
        tableName = "%s_tok" %(self.corptable)
        tokenizer = Tokenizer(use_unicode=self.use_unicode)
        self._addTransformedMessages(tableName, tokenizer.tokenize)
        
        return tableName

    def addSegmentedMessages(self, model="ctb", tmpdir="/tmp"):
        """Exports the messages to a csv, writes to a tmp file, segments, cleans up and reimports as JSON list

        Parameters
        -------
        model : :obj:`str`, optional
            model used for segmentation: ctb (Penn Chinese Treebank) or pku (Beijing Univ.)
        tmpdir : :obj:`dict`, optional
            temp directory for storing intermediate results
        """

        assert model.lower() in ["ctb", "pku"], "Available models for segmentation are CTB or PKU"
        sql = "select %s, %s from %s" % (self.messageid_field, self.message_field, self.corptable)
        rows = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        tmpfile = tmpdir+"/tmpChineseUnsegmented.txt"
        tmpfile_seg = tmpdir+"/tmpChineseSegmented.txt"

        with open(tmpfile, "w+") as a:
            w = csv.writer(a)
            w.writerows(rows)

        os.system("%s %s %s UTF-8 0 > %s" % (dlac.DEF_STANFORD_SEGMENTER , model.lower(), tmpfile, tmpfile_seg))

        new_rows = []
        raw = []
        with open(tmpfile_seg, "r") as b:
            raw = [r for r in b]
        try:
            r = csv.reader(raw)
            new_rows = [i for i in r]
        except:
            new_rows = []
            for row in raw:
                r = csv.reader([row])
                new_rows.append(next(r))

        new_rows = [[i[0].strip().replace(" ",""), # Message_ids shouldn't get split
                     i[1].strip().replace("[ ","[").replace(" ]", "]").replace(" http : //", " http://").replace(" https : //", " https://")]
                    for i in new_rows[:]]

        # os.system("rm %s %s" % (tmpfile, tmpfile_seg))

        new_rows = [(i[0], json.dumps(i[1].split(" "))) for i in new_rows[:]]

        # Now that we have the new rows, we should insert them. 1) Create table, 2) insert
        tableName = self.corptable+"_seg"
        sql = "SELECT column_name, column_type FROM INFORMATION_SCHEMA.COLUMNS "
        sql += "WHERE table_name = '%s' AND COLUMN_NAME in ('%s', '%s') and table_schema = '%s'" % (
            self.corptable, self.message_field, self.messageid_field, self.corpdb)
        types = {k:v for k,v in mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)}
        sql2 = "CREATE TABLE %s (" % (self.corptable+"_seg")
        sql2 += "%s %s primary key, %s %s " % (self.messageid_field,
                                                 types[self.messageid_field],
                                                 self.message_field,
                                                 types[self.message_field],
                                                 )
        sql2 += ")"
        mm.execute(self.corpdb, self.dbCursor, "drop table if exists "+tableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        mm.execute(self.corpdb, self.dbCursor, sql2, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        mm.standardizeTable(self.corpdb, self.dbCursor, tableName, collate=dlac.DEF_COLLATIONS[self.encoding.lower()], engine=dlac.DEF_MYSQL_ENGINE, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        alter = """ALTER TABLE %s MODIFY %s LONGTEXT""" % (tableName, self.message_field)
        mm.execute(self.corpdb, self.dbCursor, alter, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        sql = "INSERT INTO %s " % (tableName)
        sql += " VALUES (%s, %s)"
        N = dlac.MYSQL_BATCH_INSERT_SIZE
        totalLength = len(new_rows)
        for l in range(0, totalLength, N):
            print("Inserting rows (%5.2f%% done)" % (float(min(l+N,totalLength))*100/totalLength))
            mm.executeWriteMany(self.corpdb, self.dbCursor, sql, new_rows[l:l+N], writeCursor=self.dbConn.cursor(), charset=self.encoding, mysql_config_file=self.mysql_config_file)

    def addTweetPOSMessages(self):
        """Creates a POS tagged (by TweetNLP) version of the message table

        Returns
        -------
        tableName : str
            Name of POS message table: corptable_tweetpos
        """
        tableName = "%s_tweetpos" %(self.corptable)
        try:
            tagger = TweetNLP()
        except NameError:
            dlac.warn("Method not available without TweetNLP interface")
            raise
        self._addTransformedMessages(tableName, tagger.tag)

        
        return tableName

    def addTweetTokenizedMessages(self):
        """Creates a tokenized (by TweetNLP) version of the message table

        Returns
        -------
        tableName : str
            Name of tokenized message table: corptable_tweettok
        """
        tableName = "%s_tweettok" %(self.corptable)
        try:
            tokenizer = TweetNLP()
        except NameError:
            dlac.warn("Method not available without TweetNLP interface")
            raise
        self._addTransformedMessages(tableName, tokenizer.tokenize)

        return tableName

    def addSentTokenizedMessages(self, sentPerRow = False, cleanMessages = None, newlinesToPeriods=True):
        """Creates a sentence tokenized version of message table

        Returns
        -------
        tableName : str
            Name of sentence tokenized message table: corptable_stoks or corptable_sent.
        """
        if sentPerRow:
            tableName = "%s_sent" % (self.corptable)
        else:
            tableName = "%s_stoks" % (self.corptable)
        sentDetector = nltk.data.load('tokenizers/punkt/english.pickle')

        #Create Table:
        if sentPerRow:
            modify = ''#don't modify message_column
            modify_id = 'VARCHAR(128)'#modify id column
        else:
            modify = ''
            modify_id = ''
        columnNames, messageIndex, messageIdIndex = self._createTable(tableName, modify, modify_id)
                                                                    
        if cleanMessages:

            ### get lexical normalization dictionaries
            normalizeDict = {}
            
            # Han, Bo  and  Cook, Paul  and  Baldwin, Timothy, 2012
            sql = """select word, norm from %s.%s""" % (dlac.DEF_LEXICON_DB, "han_bo_emnlp_dict")
            normalizeDict.update(dict(list(mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file))))
            
            # Fei Liu, Fuliang Weng, Bingqing Wang, Yang Liu, 2011 
            # Fei Liu, Fuliang Weng, Xiao Jiang, 2012
            sql = """select word, norm from %s.%s""" % (dlac.DEF_LEXICON_DB, "liu_weng_test_set_3802")
            normalizeDict.update(dict(list(mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file))))
        

        #find all groups
        cfRows = self._findAllGroups()

        #iterate through groups in chunks
        groupsWritten = 0
        for groups in dlac.chunks(cfRows, self.groupsAtTime):
            sentRows = list()
            
            #get msgs for groups:
            rows = self._getMsgsForGroups(groups, columnNames, messageIndex)
            messages = [r[messageIndex] for r in rows]

            if messages:
                insert_idx_start = 0
                insert_idx_end = dlac.MYSQL_BATCH_INSERT_SIZE
                #tokenize msgs:
                # parses = map(lambda m: json.dumps(sentDetector.tokenize(tc.removeNonAscii(tc.treatNewlines(m.strip())))), messages)
                parses = []
                for m in messages:
                    parse = m.strip()
                    if cleanMessages:
                        parse = tc.sentenceNormalization(parse, normalizeDict, self.use_unicode)
                    parse = tc.removeNonUTF8(tc.treatNewlines(parse))
                    if newlinesToPeriods:
                        parse = parse.replace('<NEWLINE>', '.')
                    parse = sentDetector.tokenize((parse))
                    parses.append(json.dumps(parse))
                # if self.use_unicode:
                #     if cleanMessages:
                #         parses = [json.dumps(sentDetector.tokenize((tc.sentenceNormalization(m.strip(), normalizeDict, self.use_unicode)))) for m in messages]
                #     else:
                #         parses = [json.dumps(sentDetector.tokenize(tc.removeNonUTF8(tc.treatNewlines(m.strip())))) for m in messages]
                #         #parses = [json.dumps(sentDetector.tokenize(tc.removeNonUTF8(m.strip()))) for m in messages]
                # else:
                #     parses = [json.dumps(sentDetector.tokenize(tc.removeNonUTF8(tc.treatNewlines(m.strip())))) for m in messages]
                #     #parses = [json.dumps(sentDetector.tokenize(tc.removeNonUTF8(m.strip()))) for m in messages]
                #     #parses = [json.dumps(sentDetector.tokenize(tc.removeNonAscii(tc.treatNewlines(m.strip())))) for m in messages]

                #add msgs into new tables
                sql = """INSERT INTO """+tableName+""" ("""+', '.join(columnNames)+\
                        """) VALUES ("""  +", ".join(['%s']*len(columnNames)) + """)"""
                for i in range(len(rows)):
                    rows[i] = list(rows[i])
                    if sentPerRow:
                        for j, parse in enumerate(ast.literal_eval(parses[i]), 1):
                            sentRows.append(list(rows[i]))
                            sentRows[-1][messageIdIndex] = str(rows[i][messageIdIndex]) + "_" + str(j).zfill(2)
                            sentRows[-1][messageIndex] = parse
                    elif i < len(parses):
                        sentRows.append(rows[i])#debug: take out copy if eveyrthing ok to run faster
                        sentRows[i][messageIndex] = str(parses[i])
                    else:
                        dlac.warn("   warning: row %d: %s has no parse; last parse %d: %s" % (i, str(rows[i]), len(parses) - 1, str(parses[-1])))

                        # for s in range(len(rows)):#DEBUG: add back in "copy" row above to see differences in original versus parse
                        #     print("\n%d; mid: %s"%(s, str(rows[s][messageIDIndex])))
                        #     print(rows[s][messageIndex])
                        #     try: 
                        #         print(parses[s])
                        #     except IndexError:
                        #         print("!NO PARSE!")
                        # sys.exit(1)

                while insert_idx_start < len(sentRows):
                    dataToWrite = sentRows[insert_idx_start:min(insert_idx_end, len(sentRows))]
                    #dlac.warn("Inserting rows %d to %d... " % (insert_idx_start, insert_idx_end))
                    mm.executeWriteMany(self.corpdb, self.dbCursor, sql, dataToWrite, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
                    insert_idx_start += dlac.MYSQL_BATCH_INSERT_SIZE
                    insert_idx_end += dlac.MYSQL_BATCH_INSERT_SIZE
                
                groupsWritten += self.groupsAtTime
                if groupsWritten % 10000 == 0:
                    dlac.warn("  %.1fk %ss' messages sent tokenized and written" % (groupsWritten/float(1000), self.correl_field))
            else:
                dlac.warn("   Warning: No messages for:" + str(groups))

        #re-enable keys:
        mm.enableTableKeys(self.corpdb, self.dbCursor, tableName, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        return tableName

    def printTokenizedLines(self, filename, whiteListFeatTable = None):
        """Prints tokenized messages in format mallet can use

        Parameters
        ----------
        filename : str
            name of file to print to.
        whiteListFeatTable : :obj:`str`, optional
            name of white list feature table.
        """
        imp.reload(sys)
        if not self.use_unicode: sys.setdefaultencoding('utf8')
        sql = """SELECT %s, %s  from %s""" % (self.messageid_field, self.message_field,self.corptable+'_tok')
        messagesEnc = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
        try:
            messagesTok = [(m[0], json.loads(m[1])) for m in messagesEnc]
        except ValueError:
            raise ValueError("One of the tokenized messages was badly JSON encoded, please check your data again. (Maybe MySQL truncated the data?)")

        whiteSet = None
        if whiteListFeatTable:
            sql = "SELECT distinct feat FROM %s " % whiteListFeatTable[0]
            whiteSet = set([s[0] for s in mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)])

        f = open(filename, 'w')
        for m in messagesTok:
            toks = m[1]
            if whiteSet:
                toks = [w for w in toks if w in whiteSet]
            f.write("""%s en %s\n""" %(m[0], ' '.join([s for s in toks])))
        f.close()
        dlac.warn("Wrote tokenized file to: %s"%filename)

    def parseAndWriteMessages(self, sp, tableNames, messages, messageIndex, columnNames, rows):
        """Parses, then write messages, used for parallelizing parsing

        Parameters
        -------
        sp : StanfordParser object
            ?????
        tableNames : dict
            {parse type: tablename}
        messages : list
            messages to be parsed
        messageIndex : int
            index of message field
        columnNames : list
            column names in table
        rows : list
            complete row from mysql table

        Returns
        -------
        True
        """
        parses = sp.parse(messages)
        #add msgs into new tables
        for pt, tableName in list(tableNames.items()):
            sql = """REPLACE INTO """+tableName+""" ("""+', '.join(columnNames)+\
                """) VALUES ("""  +", ".join(['%s']*len(columnNames)) + """)"""
            for i in range(len(rows)):
                rows[i] = list(rows[i])
                rows[i][messageIndex] = str(parses[i][pt])

            mm.executeWriteMany(self.corpdb, self.dbCursor, sql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding, mysql_config_file=self.mysql_config_file)
        return True
    
    def addParsedMessages(self):
        """Creates a parsed version of the message table

        Returns
        -------
        tableNames : dict
            Dictionary of table names: {"pos": corptable_pos, "const": corptable_const, "dep": corptable_dep}
        """
        parseTypes = ['pos', 'const', 'dep']
        tableNames = dict( [(t, "%s_%s" %(self.corptable, t)) for t in parseTypes] )

        #Create Tables: (TODO make continue)
        for t, name in list(tableNames.items()):
            sql = "CREATE TABLE IF NOT EXISTS %s like %s" % (name, self.corptable)
            mm.execute(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            mm.standardizeTable(self.corpdb, self.dbCursor, name, collate=dlac.DEF_COLLATIONS[self.encoding.lower()], engine=dlac.DEF_MYSQL_ENGINE, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)
            mm.enableTableKeys(self.corpdb, self.dbCursor, name, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)#just incase interrupted, so we can find un-parsed groups

        #Find column names:
        columnNames = list(mm.getTableColumnNameTypes(self.corpdb, self.dbCursor, self.corptable, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file).keys())
        messageIndex = columnNames.index(self.message_field)

        #find if parsed table already has rows:
        countsql = """SELECT count(*) FROM %s""" % (tableNames[parseTypes[0]])
        cnt = mm.executeGetList(self.corpdb, self.dbCursor, countsql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)[0][0]

        #find all groups that are not already inserted
        usql = """SELECT %s FROM %s GROUP BY %s""" % (self.correl_field, self.corptable, self.correl_field)
        if cnt: #limit to those not done yet:
            usql = """SELECT a.%s FROM %s AS a LEFT JOIN %s AS b ON a.%s = b.%s WHERE b.%s IS NULL group by a.%s""" % (
                self.correl_field, self.corptable, tableNames[parseTypes[0]], self.messageid_field, self.messageid_field, self.messageid_field, self.correl_field)
        #msgs = 0#keeps track of the number of messages read
        cfRows = [r[0] for r in mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)]
        dlac.warn("parsing messages for %d '%s's"%(len(cfRows), self.correl_field))

        #disable keys (waited until after finding groups)
        for t, name in list(tableNames.items()):
            mm.disableTableKeys(self.corpdb, self.dbCursor, name, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        #iterate through groups in chunks
        if not any(field in self.correl_field.lower() for field in ["mess", "msg"]) or self.correl_field.lower().startswith("id"):
            self.groupsAtTime = 10 # if user ids
            dlac.warn("""Parsing at the non-message level is not recommended. Please rerun at message level""", attention=True)

        psAtTime = dlac.CORES / 4

        try:
            sp = StanfordParser()
        except NameError:
            dlac.warn("Method not available without StanfordParser interface")
            raise
        groupsWritten = 0
        activePs = set()
        for groups in dlac.chunks(cfRows, self.groupsAtTime):
            #get msgs for groups:
            rows = self._getMsgsForGroups(groups, columnNames, messageIndex)
            rows = [row for row in rows if row[messageIndex] and not row[messageIndex].isspace()]
            messages = [r[messageIndex] for r in rows]
            messages = [m if m else '_' for m in messages]

            #check for limit of active processes
            while (len(activePs) >= psAtTime):
                time.sleep(10)
                toRemove = set()
                for proc in activePs:
                    if not proc.is_alive():
                        toRemove.add(proc)
                        proc.join()
                        groupsWritten += self.groupsAtTime
                        if groupsWritten % 200 == 0:
                            dlac.warn("  %.1fk %ss' messages parsed and written" % (groupsWritten/float(1000), self.correl_field))
                for proc in toRemove:
                    activePs.remove(proc)
                    dlac.warn (" %s removed. (processes running: %d)" % (str(proc), len(activePs)) )


            groupsWritten += self.groupsAtTime
            if groupsWritten % 200 == 0:
                dlac.warn("  %.1fk %ss' messages parsed and written" % (groupsWritten/float(1000), self.correl_field))

            #parse msgs
            p = multiprocessing.Process(target=MessageTransformer.parseAndWriteMessages, args=(self, sp, tableNames, messages, messageIndex, columnNames, rows))
            dlac.warn (" %s starting. (processes previously running: %d)" % (str(p), len(activePs)) )
            p.start()
            activePs.add(p)

        #wait for remaining processes
        temp = activePs.copy()
        for proc in temp:
            activePs.remove(proc)
            proc.join()

        #re-enable keys:
        for t, name in list(tableNames.items()):
            mm.enableTableKeys(self.corpdb, self.dbCursor, name, charset=self.encoding, use_unicode=self.use_unicode, mysql_config_file=self.mysql_config_file)

        return tableNames
