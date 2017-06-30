import re
import sys
from html.parser import HTMLParser
from pprint import pprint

#infrastructure
from .dlaWorker import DLAWorker
from . import textCleaner as tc
from . import dlaConstants as dlac
from .mysqlMethods import mysqlMethods as mm
from .lib.happierfuntokenizing import Tokenizer #Potts tokenizer

try:
    from langid.langid import LanguageIdentifier, model
except ImportError:
    dlac.warn("Cannot import langid (cannot use addLanguageFilterTable)")
    pass

class MessageAnnotator(DLAWorker):
    """Deals with filtering or adding columns to message tables.

    Returns
    -------
    MessageAnnotator object

    Examples
    --------

    """

    def addDedupFilterTable(self, anonymize=True):
        """
        Groups all messages in a given table and filters deplicate messages within the correl_field. Writes
        a new message table called corptable_dedup. Deduplication written by Daniel Preotiuc-Pietro and adapted for DLATK.
        Removal of urls, punctuation, etc. taken from the twokenize tokenizer written by Brendan O'connor.
        """
        tokenizer = Tokenizer(use_unicode=self.use_unicode)

        new_table = self.corptable + "_dedup"
        drop = """DROP TABLE IF EXISTS %s""" % (new_table)
        create = """CREATE TABLE %s like %s""" % (new_table, self.corptable)
        mm.execute(self.corpdb, self.dbCursor, drop, charset=self.encoding, use_unicode=self.use_unicode)
        mm.execute(self.corpdb, self.dbCursor, create, charset=self.encoding, use_unicode=self.use_unicode)
        mm.standardizeTable(self.corpdb, self.dbCursor, new_table, collate=dlac.DEF_COLLATIONS[self.encoding.lower()], engine=dlac.DEF_MYSQL_ENGINE, charset=self.encoding, use_unicode=self.use_unicode)

        #Find column names:
        columnNames = list(mm.getTableColumnNameTypes(self.corpdb, self.dbCursor, self.corptable, charset=self.encoding, use_unicode=self.use_unicode).keys())
        messageIndex = columnNames.index(self.message_field)
        try:
            retweetedStatusIdx = columnNames.index("retweeted_status_text")
        except:
            retweetedStatusIdx = None
            pass

        #find all groups that are not already inserted
        usql = """SELECT %s FROM %s GROUP BY %s""" % (self.correl_field, self.corptable, self.correl_field)
        cfRows = [r[0] for r in mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode)]
        dlac.warn("deduplicating messages for %d '%s's"%(len(cfRows), self.correl_field))

        # if message level analysis
        if any(field in self.correl_field.lower() for field in ["mess", "msg"]) or self.correl_field.lower().startswith("id"):
            groupsAtTime = 1
            dlac.warn("""Deduplicating only works with a non-message level grouping such as users or counties.""", attention=True)
            sys.exit()

        groupsAtTime = 1
        rows_to_write = []
        counter = 1
        for groups in dlac.chunks(cfRows, groupsAtTime):

            # get msgs for groups:
            sql = """SELECT %s from %s where %s IN ('%s')""" % (','.join(columnNames), self.corptable, self.correl_field, "','".join(str(g) for g in groups))
            rows = list(mm.executeGetList(self.corpdb, self.dbCursor, sql, warnQuery=False, charset=self.encoding, use_unicode=self.use_unicode))
            rows = [row for row in rows if row[messageIndex] and not row[messageIndex].isspace()]

            bf = []
            for row in rows:
                row = list(row)
                try:
                    message = row[messageIndex]
                    textrt = ''
                    if retweetedStatusIdx:
                        textrt = row[retweetedStatusIdx]
                    if not textrt:
                        textrt = tc.rttext(message)[0]
                    if not textrt == '':
                        message = textrt
                        continue
                    #words = tokenizer.tokenize(message)
                    words = [word for word in tokenizer.tokenize(message) if ((word[0]!='#') and (word[0]!='@') and not tc.Exclude_RE.search(word))]

                    message = ' '.join(words).lower()
                    if len(words)>=6:
                        if 'YouTube' in words:
                            message = ' '.join(words[0:5])
                        else:
                            message = ' '.join(words[0:5])
                    if message not in bf:
                        bf.append(message)
                        if anonymize:
                            message = row[messageIndex]
                            message = tc.replaceURL(message)
                            row[messageIndex] = message
                        rows_to_write.append(row)
                    else:
                        pass
                except:
                    continue
            if len(rows_to_write) >= dlac.MYSQL_BATCH_INSERT_SIZE:
                sql = """INSERT INTO """+new_table+""" ("""+', '.join(columnNames)+""") VALUES ("""  +", ".join(['%s']*len(columnNames)) + """)"""
                mm.executeWriteMany(self.corpdb, self.dbCursor, sql, rows_to_write, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                rows_to_write = []

            if (counter % 500 == 0):
                print('%d deduplicated users inserted!' % (counter))
            counter += 1

        if rows_to_write:
            sql = """INSERT INTO """+new_table+""" ("""+', '.join(columnNames)+""") VALUES ("""  +", ".join(['%s']*len(columnNames)) + """)"""
            mm.executeWriteMany(self.corpdb, self.dbCursor, sql, rows_to_write, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

    def addAnonymizedTable(self):
        """
        
        """
        new_table = self.corptable + "_an"
        drop = """DROP TABLE IF EXISTS %s""" % (new_table)
        create = """CREATE TABLE %s like %s""" % (new_table, self.corptable)
        mm.execute(self.corpdb, self.dbCursor, drop, charset=self.encoding, use_unicode=self.use_unicode)
        mm.execute(self.corpdb, self.dbCursor, create, charset=self.encoding, use_unicode=self.use_unicode)
        mm.standardizeTable(self.corpdb, self.dbCursor, new_table, collate=dlac.DEF_COLLATIONS[self.encoding.lower()], engine=dlac.DEF_MYSQL_ENGINE, charset=self.encoding, use_unicode=self.use_unicode)

        #Find column names:
        columnNames = list(mm.getTableColumnNameTypes(self.corpdb, self.dbCursor, self.corptable, charset=self.encoding, use_unicode=self.use_unicode).keys())
        messageIndex = columnNames.index(self.message_field)
        try:
            retweetedStatusIdx = columnNames.index("retweeted_status_text")
        except:
            retweetedStatusIdx = None
            pass

        #find all groups that are not already inserted
        usql = """SELECT %s FROM %s GROUP BY %s""" % (self.correl_field, self.corptable, self.correl_field)
        cfRows = [r[0] for r in mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode)]
        dlac.warn("anonymizing messages for %d '%s's"%(len(cfRows), self.correl_field))

        groupsAtTime = 1
        rows_to_write = []
        counter = 1
        for groups in dlac.chunks(cfRows, groupsAtTime):

            # get msgs for groups:
            sql = """SELECT %s from %s where %s IN ('%s')""" % (','.join(columnNames), self.corptable, self.correl_field, "','".join(str(g) for g in groups))
            rows = list(mm.executeGetList(self.corpdb, self.dbCursor, sql, warnQuery=False, charset=self.encoding, use_unicode=self.use_unicode))
            rows = [row for row in rows if row[messageIndex] and not row[messageIndex].isspace()]

            for row in rows:
                row = list(row)
                try:
                    message = row[messageIndex]
                    message = tc.replaceUser(message)
                    message = tc.replaceURL(message)
                    row[messageIndex] = message
                    rows_to_write.append(row)
                except:
                    continue
            if len(rows_to_write) >= dlac.MYSQL_BATCH_INSERT_SIZE:
                sql = """INSERT INTO """+new_table+""" ("""+', '.join(columnNames)+""") VALUES ("""  +", ".join(['%s']*len(columnNames)) + """)"""
                mm.executeWriteMany(self.corpdb, self.dbCursor, sql, rows_to_write, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                rows_to_write = []

            if (counter % 500 == 0):
                print('%d anonymized messages inserted!' % (counter))
            counter += 1

        if rows_to_write:
            sql = """INSERT INTO """+new_table+""" ("""+', '.join(columnNames)+""") VALUES ("""  +", ".join(['%s']*len(columnNames)) + """)"""
            mm.executeWriteMany(self.corpdb, self.dbCursor, sql, rows_to_write, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)


    def addSpamFilterTable(self, threshold=dlac.DEF_SPAM_FILTER):
        """
        Groups all messages in a given table and filters spam messages within the correl_field. Writes
        a new message table called corptable_nospam with additional integer column is_spam (0 = not spam, 1 = spam).
        Spam words = 'share', 'win', 'check', 'enter', 'products', 'awesome', 'prize', 'sweeps', 'bonus', 'gift'

        Parameters
        ----------
        threshold : float
            percentage of spam messages to be considered a spam user
        """
        spam_words = ['share', 'win', 'check', 'enter', 'products', 'awesome', 'prize', 'sweeps', 'bonus', 'gift']

        new_table = self.corptable + "_nospam"
        drop = """DROP TABLE IF EXISTS %s""" % (new_table)
        create = """CREATE TABLE %s like %s""" % (new_table, self.corptable)
        add_colum = """ALTER TABLE %s ADD COLUMN is_spam INT(2) NULL""" % (new_table)
        mm.execute(self.corpdb, self.dbCursor, drop, charset=self.encoding, use_unicode=self.use_unicode)
        mm.execute(self.corpdb, self.dbCursor, create, charset=self.encoding, use_unicode=self.use_unicode)
        mm.execute(self.corpdb, self.dbCursor, add_colum, charset=self.encoding, use_unicode=self.use_unicode)
        mm.standardizeTable(self.corpdb, self.dbCursor, new_table, collate=dlac.DEF_COLLATIONS[self.encoding.lower()], engine=dlac.DEF_MYSQL_ENGINE, charset=self.encoding, use_unicode=self.use_unicode)

        #Find column names:
        columnNames = list(mm.getTableColumnNameTypes(self.corpdb, self.dbCursor, self.corptable, charset=self.encoding, use_unicode=self.use_unicode).keys())
        insertColumnNames = columnNames + ['is_spam']
        messageIndex = columnNames.index(self.message_field)

        #find all groups that are not already inserted
        usql = """SELECT %s FROM %s GROUP BY %s""" % (self.correl_field, self.corptable, self.correl_field)
        cfRows = [r[0] for r in mm.executeGetList(self.corpdb, self.dbCursor, usql, charset=self.encoding, use_unicode=self.use_unicode)]
        dlac.warn("Removing spam messages for %d '%s's"%(len(cfRows), self.correl_field))

        # if message level analysis
        if any(field in self.correl_field.lower() for field in ["mess", "msg"]) or self.correl_field.lower().startswith("id"):
            groupsAtTime = 1
            dlac.warn("""This will remove any messages that contain *any* spam words. Consider rerunning at the user level.""", attention=True)


        groupsAtTime = 1
        rows_to_write = []
        counter = 1
        users_removed = 0
        for groups in dlac.chunks(cfRows, groupsAtTime):

            # get msgs for groups:
            sql = """SELECT %s from %s where %s IN ('%s')""" % (','.join(columnNames), self.corptable, self.correl_field, "','".join(str(g) for g in groups))
            rows = list(mm.executeGetList(self.corpdb, self.dbCursor, sql, warnQuery=False, charset=self.encoding, use_unicode=self.use_unicode))
            rows = [row for row in rows if row[messageIndex] and not row[messageIndex].isspace()]

            total_messages = float(len(rows))
            if total_messages == 0: continue
            spam_messages = 0
            insert_rows = []
            for row in rows:
                try:
                    if any(word in row[messageIndex] for word in spam_words):
                        spam_messages += 1
                        insert_rows.append(row + (1,))
                    else:
                        insert_rows.append(row + (0,))
                except:
                    continue

            if spam_messages/total_messages < threshold:
                rows_to_write += insert_rows
            else:
                users_removed += 1

            if len(rows_to_write) >= dlac.MYSQL_BATCH_INSERT_SIZE:
                sql = """INSERT INTO """+new_table+""" ("""+', '.join(insertColumnNames)+""") VALUES ("""  +", ".join(['%s']*len(insertColumnNames)) + """)"""
                mm.executeWriteMany(self.corpdb, self.dbCursor, sql, rows_to_write, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                rows_to_write = []

            if (counter % 500 == 0):
                print('%d users filtered!' % (counter))
            counter += 1

        if rows_to_write:
            sql = """INSERT INTO """+new_table+""" ("""+', '.join(insertColumnNames)+""") VALUES ("""  +", ".join(['%s']*len(insertColumnNames)) + """)"""
            mm.executeWriteMany(self.corpdb, self.dbCursor, sql, rows_to_write, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
        print('%d users removed!' % (users_removed))

    # TODO: add nicer implementation
    def yieldMessages(self, messageTable, totalcount):
        if totalcount > 10*dlac.MAX_SQL_SELECT:
            for i in range(0,totalcount, dlac.MAX_SQL_SELECT):
                sql = "SELECT * FROM %s limit %d, %d" % (messageTable, i, dlac.MAX_SQL_SELECT)
                for m in mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode):
                    yield [i for i in m]
        else:
            sql = "SELECT * FROM %s" % messageTable
            for m in mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode):
                yield m

    def addLanguageFilterTable(self, langs, cleanMessages, lowercase):
        """Filters all messages in corptable for a given language. Keeps messages if
        confidence is greater than 80%. Uses the langid library.

        Parameters
        ----------
        langs : list
            list of languages to filter for
        cleanMessages : boolean
            remove URLs, hashtags and @ mentions from messages before running langid
        lowercase : boolean
            convert message to all lowercase before running langid
        """
        identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

        new_table = self.corptable + "_%s"

        columnNames = mm.getTableColumnNames(self.corpdb, self.corptable, charset=self.encoding, use_unicode=self.use_unicode)
        messageIndex = [i for i, col in enumerate(columnNames) if col.lower() == dlac.DEF_MESSAGE_FIELD.lower()][0]
        #messageIDindex = [i for i, col in enumerate(columnNames) if col.lower() == dlac.DEF_MESSAGEID_FIELD.lower()][0]

        # CREATE NEW TABLES IF NEEDED
        messageTables = {l: new_table % l for l in langs}
        for l, table in messageTables.items():
            drop = """DROP TABLE IF EXISTS %s""" % (table)
            create = """CREATE TABLE %s like %s""" % (table, self.corptable)
            mm.execute(self.corpdb, self.dbCursor, drop, charset=self.encoding, use_unicode=self.use_unicode)
            mm.execute(self.corpdb, self.dbCursor, create, charset=self.encoding, use_unicode=self.use_unicode)
            mm.standardizeTable(self.corpdb, self.dbCursor, table, collate=dlac.DEF_COLLATIONS[self.encoding.lower()], engine=dlac.DEF_MYSQL_ENGINE, charset=self.encoding, use_unicode=self.use_unicode)

        #ITERATE THROUGH EACH MESSAGE WRITING THOSE THAT ARE ENGLISH
        messageDataToAdd = {l: list() for l in langs}
        messageDataCounts = {l: 0 for l in langs}
        totalMessages = 0
        totalMessagesKept = 0
        sql = """SELECT COUNT(*) FROM %s""" % self.corptable
        totalMessagesInTable = mm.executeGetList(self.corpdb, self.dbCursor, sql)[0][0]

        print("Reading %s messages" % ",".join([str(totalMessagesInTable)[::-1][i:i+3] for i in range(0,len(str(totalMessagesInTable)),3)])[::-1])
        memory_limit = dlac.MYSQL_BATCH_INSERT_SIZE if dlac.MYSQL_BATCH_INSERT_SIZE < totalMessagesInTable else totalMessagesInTable/20

        html = HTMLParser()
        for messageRow in self.yieldMessages(self.corptable, totalMessagesInTable):
            messageRow = list(messageRow)
            totalMessages+=1
            message = messageRow[messageIndex]

            try:
                message = message.encode('utf-8', 'ignore').decode('windows-1252', 'ignore')
            except UnicodeEncodeError as e:
                raise ValueError("UnicodeEncodeError"+ str(e) + str([message]))
            except UnicodeDecodeError as e:
                print(type(message))
                print([message.decode('utf-8')])
                raise ValueError("UnicodeDecodeError"+ str(e) + str([message]))
            except AttributeError as e:
                print("Empty message, skipped")
                continue
            try:
                message = html.unescape(message)
                messageRow[messageIndex] = message

                if cleanMessages:
                    message = re.sub(r"(?:\#|\@|https?\://)\S+", "", message)
            except Exception as e:
                print(e)
                print([message])

            try:
                if lowercase:
                    message = message.lower()
            except Exception as e:
                print(e)
                print([message])


            lang, conf = None, None
            try:
                lang, conf = identifier.classify(message)
            except TypeError as e:
                print(("         Error, ignoring row %s" % str(messageRow)))

            if lang in langs and conf > .80 :
                messageDataToAdd[lang].append(messageRow)
                messageDataCounts[lang] += 1

                totalMessagesKept+=1
            else:
                continue

            if totalMessagesKept % memory_limit == 0:
                #write messages every so often to clear memory
                for l, messageData in messageDataToAdd.items():
                    sql = """INSERT INTO """+messageTables[l]+""" ("""+', '.join(columnNames)+""") VALUES ("""  +", ".join(['%s']*len(columnNames)) + """)"""
                    mm.executeWriteMany(self.corpdb, self.dbCursor, sql, messageData, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)
                    messageDataToAdd[l] = list()

                for l, nb in [(x[0], len(x[1])) for x in iter(messageDataToAdd.items())]:
                    messageDataCounts[l] += nb

                print("  %6d rows written (%6.3f %% done)" % (totalMessagesKept,
                                                              100*float(totalMessages)/totalMessagesInTable))

        if messageDataToAdd:
            print("Adding final rows")
            for l, messageData in messageDataToAdd.items():
                sql = """INSERT INTO """+messageTables[l]+""" ("""+', '.join(columnNames)+""") VALUES ("""  +", ".join(['%s']*len(columnNames)) + """)"""
                mm.executeWriteMany(self.corpdb, self.dbCursor, sql, messageData, writeCursor=self.dbConn.cursor(), charset=self.encoding, use_unicode=self.use_unicode)

        print("Kept %d out of %d messages" % (totalMessagesKept, totalMessages))
        pprint({messageTables[l]: v for l, v in messageDataCounts.items()})