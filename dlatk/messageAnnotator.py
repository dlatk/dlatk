import re
import sys
import html
from pprint import pprint

#infrastructure
from .database.query import Column
from .dlaWorker import DLAWorker
from . import textCleaner as tc
from . import dlaConstants as dlac
from .lib.happierfuntokenizing import Tokenizer #Potts tokenizer

try:
    from langid.langid import LanguageIdentifier, model
except ImportError:
    dlac.warn("Warning: Cannot import langid (cannot use addLanguageFilterTable)")
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

        # Drop if the table exists, create a new table, and standardize it.
        dropQuery = self.qb.create_drop_query(new_table)
        dropQuery.execute_query()

        createQuery = self.qb.create_createTable_query(new_table).like(self.corptable)
        createQuery.execute_query()

        self.data_engine.standardizeTable(
            new_table, 
            collate=dlac.DEF_COLLATIONS[self.encoding.lower()], 
            engine=dlac.DEF_MYSQL_ENGINE, 
            charset=self.encoding, 
            use_unicode=self.use_unicode)

        #Find column names:
        columnNames = list(self.data_engine.getTableColumnNameTypes(self.corptable))
        messageIndex = columnNames.index(self.message_field)
        try:
            retweetedStatusIdx = columnNames.index("retweeted_status_text")
        except:
            retweetedStatusIdx = None
            pass

        #find all groups that are not already inserted
        selectQuery = self.qb.create_select_query(self.corptable).set_fields([self.correl_field]).group_by([self.correl_field])
        cfRows = [r[0] for r in selectQuery.execute_query()]
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
            where_condition = "%s IN ('%s')" % (self.correl_field, "','".join(str(g) for g in groups))
            selectQuery = self.qb.create_select_query(self.corptable).where(where_condition).set_fields(columnNames)
            rows = selectQuery.execute_query()
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
                insertQuery = self.qb.create_insert_query(new_table).set_values([(name, '') for name in columnNames])
                insertQuery.execute_query(rows_to_write)
                rows_to_write = []

            if (counter % 500 == 0):
                print('%d deduplicated users inserted!' % (counter))
            counter += 1

        if rows_to_write:
            insertQuery = self.qb.create_insert_query(new_table).set_values([(name, '') for name in columnNames])
            insertQuery.execute_query(rows_to_write)

    def addAnonymizedTable(self):
        """
        
        """
        new_table = self.corptable + "_an"
         
        # Drop if the table exists, create a new table, and standardize it.
        dropQuery = self.qb.create_drop_query(new_table)
        dropQuery.execute_query()

        createQuery = self.qb.create_createTable_query(new_table).like(self.corptable)
        createQuery.execute_query()

        self.data_engine.standardizeTable(
            new_table, 
            collate=dlac.DEF_COLLATIONS[self.encoding.lower()], 
            engine=dlac.DEF_MYSQL_ENGINE, 
            charset=self.encoding, 
            use_unicode=self.use_unicode)

        #Find column names:
        columnNames = list(self.data_engine.getTableColumnNameTypes(self.corptable))
        messageIndex = columnNames.index(self.message_field)
        try:
            retweetedStatusIdx = columnNames.index("retweeted_status_text")
        except:
            retweetedStatusIdx = None
            pass

        #find all groups that are not already inserted
        selectQuery = self.qb.create_select_query(self.corptable).set_fields([self.correl_field]).group_by([self.correl_field])
        cfRows = [r[0] for r in selectQuery.execute_query()]
        dlac.warn("anonymizing messages for %d '%s's"%(len(cfRows), self.correl_field))

        groupsAtTime = 1
        rows_to_write = []
        counter = 1
        for groups in dlac.chunks(cfRows, groupsAtTime):

            # get msgs for groups:
            where_condition = "%s IN ('%s')" % (self.correl_field, "','".join(str(g) for g in groups))
            selectQuery = self.qb.create_select_query(self.corptable).where(where_condition).set_fields(columnNames)
            rows = selectQuery.execute_query()
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
                insertQuery = self.qb.create_insert_query(new_table).set_values([(name, '') for name in columnNames])
                insertQuery.execute_query(rows_to_write)
                rows_to_write = []

            if (counter % 500 == 0):
                print('%d anonymized messages inserted!' % (counter))
            counter += 1

        if rows_to_write:
            insertQuery = self.qb.create_insert_query(new_table).set_values([(name, '') for name in columnNames])
            insertQuery.execute_query(rows_to_write)

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

        # Drop if the table exists, create a new table, add spam identifier column, and standardize it.
        dropQuery = self.qb.create_drop_query(new_table)
        dropQuery.execute_query()

        createQuery = self.qb.create_createTable_query(new_table).like(self.corptable)
        createQuery.execute_query()
        
        column = Column("is_spam", "INT(2)")
        self.qb.create_createColumn_query(new_table, column).execute_query()

        self.data_engine.standardizeTable(
            new_table, 
            collate=dlac.DEF_COLLATIONS[self.encoding.lower()], 
            engine=dlac.DEF_MYSQL_ENGINE, 
            charset=self.encoding, 
            use_unicode=self.use_unicode)

        #Find column names:
        columnNames = list(self.data_engine.getTableColumnNameTypes(self.corptable))
        insertColumnNames = columnNames + ['is_spam']
        messageIndex = columnNames.index(self.message_field)

        #find all groups that are not already inserted
        selectQuery = self.qb.create_select_query(self.corptable).set_fields([self.correl_field]).group_by([self.correl_field])
        cfRows = [r[0] for r in selectQuery.execute_query()]
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
            where_condition = "%s IN ('%s')" % (self.correl_field, "','".join(str(g) for g in groups))
            selectQuery = self.qb.create_select_query(self.corptable).where(where_condition).set_fields(columnNames)
            rows = selectQuery.execute_query()
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
                insertQuery = self.qb.create_insert_query(new_table).set_values([(name, '') for name in insertColumnNames])
                insertQuery.execute_query(rows_to_write)
                rows_to_write = []

            if (counter % 500 == 0):
                print('%d users filtered!' % (counter))
            counter += 1

        if rows_to_write:
            insertQuery = self.qb.create_insert_query(new_table).set_values([(name, '') for name in insertColumnNames])
            insertQuery.execute_query(rows_to_write)
        print('%d users removed!' % (users_removed))

    # TODO: add nicer implementation
    def yieldMessages(self, messageTable, totalcount, columnNames='*'):
        if totalcount > 10*dlac.MAX_SQL_SELECT:
            for i in range(0,totalcount, dlac.MAX_SQL_SELECT):
                selectQuery = self.qb.create_select_query(messageTable).set_fields(columnNames).set_limit("{}, {}".format(i, dlac.MAX_SQL_SELECT))
                for m in selectQuery.execute_query():
                    yield [i for i in m]
        else:
            selectQuery = self.qb.create_select_query(messageTable).set_fields(columnNames)
            for m in selectQuery.execute_query():
                yield m

    def addLanguageFilterTable(self, langs, cleanMessages, lowercase, lightEnglishFilter=False):
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
        if lightEnglishFilter: 
            identifier.set_languages(['en', 'es'])
            langs = ['en']

        new_table = self.corptable + "_%s"

        columnNames = list(self.data_engine.getTableColumnNameTypes(self.corptable))
        #print("column names", columnNames) #DEBUG
        assert len(columnNames) > 0, "no columns in message table, check database name"
        messageIndex = [i for i, col in enumerate(columnNames) if col.lower() == dlac.DEF_MESSAGE_FIELD.lower()][0]
        #messageIDindex = [i for i, col in enumerate(columnNames) if col.lower() == dlac.DEF_MESSAGEID_FIELD.lower()][0]
        #print("messageIndex", messageIndex) #DEBUG

        # CREATE NEW TABLES IF NEEDED
        messageTables = {l: new_table % l for l in langs}
        for l, table in messageTables.items():

            dropQuery = self.qb.create_drop_query(table)
            dropQuery.execute_query()

            createQuery = self.qb.create_createTable_query(table).like(self.corptable)
            createQuery.execute_query()
            
            self.data_engine.standardizeTable(
                table, 
                collate=dlac.DEF_COLLATIONS[self.encoding.lower()], 
                engine=dlac.DEF_MYSQL_ENGINE, 
                charset=self.encoding, 
                use_unicode=self.use_unicode)

        #ITERATE THROUGH EACH MESSAGE WRITING THOSE THAT ARE ENGLISH
        messageDataToAdd = {l: list() for l in langs}
        messageDataCounts = {l: 0 for l in langs}
        totalMessages = 0
        totalMessagesKept = 0
        selectQuery = self.qb.create_select_query(self.corptable).set_fields(["COUNT(*)"])
        totalMessagesInTable = selectQuery.execute_query()[0][0]

        print("Reading %s messages" % ",".join([str(totalMessagesInTable)[::-1][i:i+3] for i in range(0,len(str(totalMessagesInTable)),3)])[::-1])
        memory_limit = dlac.MYSQL_BATCH_INSERT_SIZE if dlac.MYSQL_BATCH_INSERT_SIZE < totalMessagesInTable else totalMessagesInTable/20

        for messageRow in self.yieldMessages(self.corptable, totalMessagesInTable, columnNames):
            messageRow = list(messageRow)
            totalMessages+=1
            message = messageRow[messageIndex]
            #print("message", message, messageIndex) #DEBUG
            try:
                message = tc.removeNonAscii(message)
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

            if lightEnglishFilter and ((lang == 'es' and conf < .95) or (lang == 'en' and conf > .60)):
                lang = 'en'
                messageDataToAdd[lang].append(messageRow)
                messageDataCounts[lang] += 1

                totalMessagesKept+=1

            elif lang in langs and conf > dlac.DEF_LANG_FILTER_CONF:
                messageDataToAdd[lang].append(messageRow)
                messageDataCounts[lang] += 1

                totalMessagesKept+=1

            else:
                continue

            if totalMessagesKept % memory_limit == 0:
                #write messages every so often to clear memory
                for l, messageData in messageDataToAdd.items():
                    insertQuery = self.qb.create_insert_query(messageTables[l]).set_values([(name, '') for name in columnNames])
                    insertQuery.execute_query(messageData)
                    messageDataToAdd[l] = list()

                for l, nb in [(x[0], len(x[1])) for x in iter(messageDataToAdd.items())]:
                    messageDataCounts[l] += nb

                print("  %6d rows written (%6.3f %% done)" % (totalMessagesKept,
                                                              100*float(totalMessages)/totalMessagesInTable))


        if messageDataToAdd:
            print("Adding final rows")
            for l, messageData in messageDataToAdd.items():
                insertQuery = self.qb.create_insert_query(messageTables[l]).set_values([(name, '') for name in columnNames])
                insertQuery.execute_query(messageData)

        print("Kept %d out of %d messages" % (totalMessagesKept, totalMessages))
        pprint({messageTables[l]: v for l, v in messageDataCounts.items()})
