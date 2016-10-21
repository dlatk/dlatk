#!/usr/bin/env python
#####################################################
##PYTHON SCRIPT TO INTERACT WITH PERMA LEXICON DBs
##
##
## andy.schwartz@gmail.com - Summer 2011
##
##TODO:
##-make sure printCSV covers all bases.
##-Add fucntionaltiy for adding terms to db
##-make sure add terms works with hashtags

import math
import sys,os, getpass
import re, csv
import pprint
import time
import MySQLdb
import random

from optparse import OptionParser, OptionGroup
try:
    from nltk.corpus import wordnet as wn
except ImportError:
    print 'LexInterface:warning: nltk.corpus module not imported.'


#MySQLdb.paramstyle 
  	

##CONSTANTS (STATIC VARIABLES)##
PERMA_CODES = {'P+': 'positive emotion',
              'P-': 'negative emotion',
              'E+': 'engagement',
              'E-': 'disengagement',
              'R+': 'positive relationships',
              'R-': 'negative relationships',
              'M+': 'meaning in life',
              'M-': 'meaninglessness',
              'A+': 'accomplishment',
              'A-': 'unaccomplishment'
              }

PERMA_SHORT_DEF = {'P+': 'positive feelings such as happiness, joy, excitement, enthusiasm, and contentment',
                   'P-': 'negative feelings such as distress, anger, contempt, disgust, fear, nervousness, and sadness',
                   'E+': 'being absorbed, focused, or interested in what one is doing',
                   'E-': 'withdrawal from life, lack of activity, disinterest, or boredom',
                   'R+': 'caring for others and feeling cared for, loved, esteemed, and valued by others',
                   'R-': 'disconnected from others, loneliness, or feeling like their relationships with other people are bad for them',
                   'M+': 'feeling of belonging or serving something larger than the self, having a sense of purpose or direction',
                   'M-': 'feeling disconnected, worthless, drifting, like their life is without a purpose',
                   'A+': 'achievement, success, or feeling like they mastered something',
                   'A-': 'lack of achievement or mastery, settling for status quo or mediocracy'
              }
PERMA_LONG_DEF = {'P+': 'If we placed all emotions on a spectrum ranging from pleasant to unpleasant, positive emotions are those emotions on the pleasant end that people prefer to feel (e.g., feeling grateful, upbeat; expressing appreciation, liking). Positive emotions can be intense, or fleeting (i.e. excitement, ecstasy, or joy) or they can be milder, and longer lasting (i.e. gratitude, contentment, or peaceful).',
              'P-': 'Negative emotions represent unpleasant feelings, that people would prefer not to feel (e.g., feeling contemptuous, irritable; expressing disdain, disliking). Negative emotions can be primal and intense (i.e. rage, jealousy, or depression) or they can be milder (i.e. annoyance, dislike, or sadness).',
              'E+': 'When one is engaged in their life, work, or activity, the individual is completely absorbed, interested and focused on what they are doing. The extreme form of engagement is sometimes called "flow." In flow, time passes quickly, attention is completely focused on the activity and the sense of self is lost. The message can be in the past tense .',
              'E-': 'When one is not feeling engaged in their life, work, or activity the individual can be feeling bored, uninterested, sleepy or unable to concentrate on the present moment. When rating messages, for a high rating in unengagement the message would undoubtedly be expressing strong unengagement in their activity and most of the message would be about expressing that unengagement. The message can be in the past tense.',
              'R+': 'Whe one feels that relationships are good, making them feel loved, respected and valued by the others in their lives or having positive feelings towards others in their lives.  One might express feeling grateful and loving for the people in the in their lives.',
              'R-': 'When one feels lonely, disconnected from others, judgmental of others or generally feeling like they have few or no good relationships in their lives or that the relationships. One may also express dissatisfaction with their current relationships.',
              'M+': 'When one generally feels like they have a sense of direction in their lives. One may express being "called" to do something or like they have set life plan. One might talk about their religion or spirituality or they may talk about their fate and destiny. One may also talk about particular events happening for a reason.',
              'M-': 'When one generally feels direction-less in life or like their life or life events lack any higher meaning. One may express that their lives are pointless or may talk about particular life events are pointless and server no higher reason.',
              'A+': 'When one feels like they are accomplishing something or like they are moving forward and advancing in their life. One may express feeling proud of doing something or proud of someone else\'s accomplishments.',
              'A-': 'When one feels like they are stuck, or not accomplishing anything with their life. One may express feeling disappointed with themselves or like they\'re not measuring up.'
              }


HOST = 'localhost'
USER = getpass.getuser()
DB = 'permaLexicon'

DEF_CORPDB = 'fb14'
DEF_CORPTABLE = 'messages'
DEF_TERMFIELD = 'term'
DEF_MESSAGEFIELD = 'message'
DEF_MESSAGEIDFIELD = 'message_id'
DEF_MINWORDFREQ = 1000;
DEF_NUMRANDMESSAGES = 100
MAX_WRITE_RECORDS = 1000 #maximum number of records to write at a time (for add_terms...)

############################################################
## Class / Static Methods

def warn(string):
    print >>sys.stderr, string

def loadLexiconFromFile(filename):
    """Loads the perma lexicon, using standard formatting
    returns a dictionary of frozensets"""
    lexFile = open(filename)
    reader = csv.reader(lexFile,dialect='excel')
    # comma = re.compile(r'\,\s*')
    cats = []
    lexicon = {}
    # for line in lexFile:
    for terms in reader:
        # terms = comma.split(line.rstrip())
        if len(cats) > 0:
            for i in range(len(terms)):
                if (terms[i]):
                    lexicon[cats[i]].append(terms[i])
        else:
            for i in range(len(terms)):
                lexicon[terms[i]] = []
                cats.append(terms[i])
    
    for cat in cats:
        lexicon[cat] = frozenset(lexicon[cat])

    return lexicon

#LUKASZ EDIT
def loadLexiconFromGFile(filename, using_filter):
    """Loads a lexicon in "google format"
    returns a dictionary of frozensets"""
    lexFile = open(filename)
    lexLines = lexFile.read().split('\r')
    lexLines = lexLines[1:len(lexLines)]

    comma = re.compile(r'\,')
    cats = []
    lexicon = {}
    for line in lexLines:
        line_split = comma.split(line)

        if len(line_split)>1:
            this_cat = line_split[1]
            this_term = line_split[0]
            this_keep = int(line_split[2])
            if lexicon.has_key(this_cat):
                if using_filter:
                    if this_keep == 1:
                        lexicon[this_cat].append(this_term)
                else:
                    lexicon[this_cat].append(this_term)
            else:
                lexicon[this_cat] = []
                cats.append(this_cat)

    for cat in cats:
        lexicon[cat] = frozenset(lexicon[cat])

    return lexicon    

def loadLexiconFromSparse(filename):
    """Loads the perma lexicon from a sparse formatting word[, word], category
    returns a dictionary of frozensets"""
    lexFile = open(filename)
    comma = re.compile(r'\,\s*')
    cats = []
    lexicon = {}
    for line in lexFile:
        items = comma.split(line.rstrip())
        terms = items[0:-1]
        cat = items[-1]
        if cat not in lexicon:
            lexicon[cat] = []
        for term in terms:
            if term: 
                lexicon[cat].append(term)
    
    for cat in lexicon.keys():
        lexicon[cat] = frozenset(lexicon[cat])

    return lexicon

def loadWeightedLexiconFromSparse(filename):
    """Loads the perma lexicon from a sparse formatting word[, word], category
    returns a dictionary of frozensets"""
    lexFile = open(filename)
    comma = re.compile(r'\,\s*')
    cats = [] 
    lexicon = {}
    for line in lexFile:
        items = comma.split(line.rstrip())
        terms = items[0:-2]
        cat = items[-2]
	weight = items[-1]
        if cat not in lexicon:
            lexicon[cat] = dict()
        for term in terms:
            if term:
                lexicon[cat][term]=weight

    return lexicon


def loadLexiconFromDic(filename):
    """Loads a lexicon from a .dic file such as LIWC2001_English.dic"""
    lexicon = {}

    with open(filename, 'rb') as f:
        reading_categories = False
        cat_map = {}

        for line in f.readlines():
            if line:
                line = line.strip('\r\n')
                line = line.strip()
                if not line:
                    continue
            if line[0] == '%':
                reading_categories = not reading_categories
                continue
            line_split = [l for l in line.split('\t') if l] 
            if reading_categories:
                cat_map[int(line_split[0])] = line_split[1].strip().upper()
                # the above looks like: {1:PRONOUN, 2:I, 3:WE, ...}
            else:
                word = line_split[0]
                if line_split[1][0] == '(':
                    print 'warning: line [%s] partially discarded due to inconsistent formatting'%(line, )
                    continue
                try:
                    word_categories = map(lambda x: cat_map[x], map(int, line_split[1:]))
                except KeyError as e:
                    print "Category", line_split.pop(line_split.index(str(e.args[0]))), "doesn't exist [word: %s]" % line_split[0]
                word_categories = map(lambda x: cat_map[x], map(int, line_split[1:]))
                for category in word_categories:
                    try:
                        lexicon[category].add(word)
                    except KeyError:
                        lexicon[category] = set()
                        lexicon[category].add(word)

    for category in lexicon.keys():
        lexicon[category] = frozenset(lexicon[category])

    return lexicon


def loadLexiconFeatMapFromCSV(filename):
    """Load a lexicon from a csv"""
    import csv
    csvReader = csv.reader(open(filename, 'rUb'))
    header = csvReader.next() #should be topic, topic_label, ....
    
    lexicon = {}
    labels_used = {}
    for row in csvReader:
        topic, label = row[0:2]
        original_label = label.lower()
        new_label = None
        
        if labels_used.has_key(original_label):
            labels_used[original_label] += 1
            new_label = '%s_%d'%(original_label, labels_used[original_label] + 1)

        if new_label:
            lexicon[new_label] = set()
            lexicon[new_label].add(topic)
        else:
            lexicon[original_label] = set()
            lexicon[original_label].add(topic)
            
        labels_used[original_label] = labels_used.get(original_label, 1)

        print topic, original_label, new_label

    return lexicon

def loadLexiconFromTopicFile(filename):
    """Loads a lexicon from a topic file
    returns a dictionary of frozensets"""
    lexFile = open(filename, 'rb')

    lexicon = {}
    for line in lexFile.readlines():
        line_split = line.split('\t')
        if len(line_split)>1:
            category = str(line_split[0])
            lexicon[category] = set()
            terms = line_split[2]
            term_split = terms.split(' ')
            for term in term_split:
                if term != '\n':
                    lexicon[category].add(term)

    for category in lexicon.keys():
        lexicon[category] = frozenset(lexicon[category])

    return lexicon
   
def loadWeightedLexiconFromTopicFile(filename):
    """Loads a weighted lexicon 
    returns a dictionary of dictionaries"""
    lexFile = open(filename, 'rb')

    lexicon = {}
    for line in lexFile.readlines():
        line_split = line.split('\t')
        if len(line_split)>1:
            category = str(line_split[0])
            weight = line_split[1]
            terms = line_split[2]
            lexicon[category] = {}
            term_split = terms.split(' ')
            for term in term_split:
                if term != '\n':
                    lexicon[category][term] = weight
    return lexicon
   
def loadWeightedLexiconFromTopicCSV(filename, threshold=None):
    """Loads a weighted lexicon 
    returns a dictionary of dictionaries"""
    import csv
    csvReader = csv.reader(open(filename, 'rb'))
    header = csvReader.next() #should be topic_id, word1, ...etc..
    print "Loading %s" % filename
    lexicon = {}
    for row in csvReader:
         topic = row[0]
         wordScores = row[1:]
         words = wordScores[::2]
         weights = wordScores[1::2]
         weights = map(float, weights)

         if threshold:
             new_words = []
             new_weights = []
             keep_pairs_with_weights_above_this_number = weights[0] * threshold
             if threshold == float('-inf'):
                 keep_pairs_with_weights_above_this_number = threshold
             for ii in range(len(weights)):
                 if weights[ii] > keep_pairs_with_weights_above_this_number:
                     new_words.append(words[ii])
                     new_weights.append(weights[ii])
             words = new_words
             weights = new_weights

         #now the weight for word[i] is weight[i]
         lexicon[topic] = {}
         for ii in xrange(0, len(words)):
             lexicon[topic][words[ii]] = weights[ii]
             #print >> sys.stderr, "topic: %s, word: %s, weight: %2.2f"%(topic, words[ii], weights[ii])
    print "Done, num_topics: %d" % len(lexicon)
    return lexicon

     
def dbConnect(mysql_host=HOST):
    dbConn = MySQLdb.connect (host = mysql_host,
                              read_default_file='~/.my.cnf',
                              db = DB, charset= 'utf8mb4', use_unicode = True)
    dbCursor = dbConn.cursor()
    return (dbConn, dbCursor)

def abstractDBConnect(host, user, db):
    dbConn = MySQLdb.connect (host = host,
                          user = user,
                          db = db)
    dbCursor = dbConn.cursor()
    return (dbConn, dbCursor)



def interactiveGetSenses(cat, word):
    os.system('clear')
    print "\n[%s] \033[92m%s\033[0m\n%s" %(PERMA_CODES[cat].title(), word, '='*(len(word)+20))
    print "\033[90m%s\033[0m\n"%PERMA_LONG_DEF[cat]
    currentSenses = set()
    POSs = {wn.NOUN: 'noun', wn.VERB: 'verb', wn.ADJ: 'adjective', wn.ADV: 'adverb'}
    for pos, posName in POSs.iteritems():
        synsets = wn.synsets(word, pos)
        if synsets:
            print "\t%s:" % posName
            i = 1
            wss = [None,]
            for syns in synsets:
                wss.append(syns.name+'.'+word)
                print "\t\t\033[92m %d: \033[0m(%s)\033[92m %s\033[0m" % (i, ', '.join([lemma.name for lemma in syns.lemmas]), syns.definition)
                i+=1
            answered = False
            senses = None
            while not senses: 
                
                print "\n\tWhich of the above senses expresses \033[1m%s (i.e. %s)\033[0m?" % (PERMA_CODES[cat].title(), PERMA_SHORT_DEF[cat])
                senses = raw_input("\t(separate with spaces; 0 => none; cntrl-c to quit)? ")

                #validity check:
                senses = senses.strip()
                if not re.match(r'^[0-9, ]+$', senses):
                    print "entered non-numeric character"
                    senses = None
                    continue
                senses = re.findall(r'\d+', senses)
                ins = set(range(len(wss)))
                for s in senses:
                    s = int(s)
                    if s == 0 and len(senses) > 1:
                        print "entered 0 along with other senses"
                        senses = None
                    if s not in ins:
                        print "%d not a choice" % s
                        senses = None

                #add to set:
            for s in senses:
                if s > 0:
                    ws = wss[int(s)]
                    print "\t\t\tadding %s" % ws
                    currentSenses.add(ws)

    print "The following will be added: %s" % currentSenses
    return currentSenses
                


#################################################################
## CLASS SETUP
#
class Lexicon(object):

    #instance variables:
    dbConn = None
    dbCursor = None
    currentLexicon = None
    lexiconDB = DB

    def __init__(self, lex = None, mysql_host = HOST):
        (self.dbConn, self.dbCursor) = dbConnect(mysql_host)
        self.mysql_host = mysql_host
        self.currentLexicon = lex

    def __str__(self):
        return str(self.currentLexicon)

    ############################################################
    ## Instance Methods
    #

    def createLexiconTable(self, tablename):
        """Creates a lexicon table from the instances lexicon variable"""
        
        #first create the table:
        enumCats = "'"+"', '".join(map(lambda k: k.upper().replace("'", "\\'"), self.currentLexicon.keys()))+"'"   
        drop = """DROP TABLE IF EXISTS """+tablename
        sql = """CREATE TABLE IF NOT EXISTS %s (id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                 term VARCHAR(128), category ENUM(%s), INDEX(term), INDEX(category)) CHARACTER SET utf8 COLLATE utf8_general_ci ENGINE = MyISAM""" % (tablename, enumCats)
        print "Running: ", drop
        print "and:     ", sql
        try:
            self.dbCursor.execute(drop)
            self.dbCursor.execute(sql)
        except MySQLdb.Error, e:
            warn("MYSQL ERROR in createLexiconTable" + str(e))
            sys.exit(1)

        #next insert rows:
        self.insertLexiconRows(tablename)

    def insertLexiconRows(self, tablename, lex=None):
        """Adds rows, taken from the lexicon variable to mysql"""
        if not lex: lex = self.currentLexicon
        #SETUP QUERY:
        sqlQuery = """INSERT INTO """+tablename+""" (term, category) values (%s, %s)"""
        values = []
        for cat, terms in lex.iteritems():
            values.extend(map(lambda term: [term, cat.upper()], terms))
        
        try:
            self.dbCursor.executemany(sqlQuery, values)
        except MySQLdb.Error, e:
            warn("MYSQL ERROR in insertLexiconRows:" + str(e) + sqlQuery);
            sys.exit(1)

            
    def setLexicon(self, lexicon):
        self.currentLexicon = lexicon
    def getLexicon(self, temp="nothing"):
        return self.currentLexicon

    def loadLexicon(self, tablename, where = ''):
        """Loads a lexicon as currentLexicon"""
        sqlQuery = "SELECT term, category from "+tablename
        if where:
            sqlQuery += ' WHERE ' + where
        
        data = []
#        try:
        self.dbCursor.execute(sqlQuery)
        data = self.dbCursor.fetchall()
#        except MySQLdb.Error, e:
#            warn(" MYSQL ERROR" + str(e))
#            sys.exit(1)
        lexicon = {}
        for row in data:
            if not row[1] in lexicon:
                lexicon[row[1]] = []
            lexicon[row[1]].append(row[0])

        self.currentLexicon = {}
        for cat, terms in lexicon.iteritems():
            self.currentLexicon[cat] = frozenset(terms)

        return self.currentLexicon
    
    def depolCategories(self):
        newDict = {}
        myDict = self.currentLexicon

        for cat in myDict.keys():
            newCat = cat.rstrip('+').rstrip('-')
            if not newCat in newDict:
                newDict[newCat] = []
            newDict[newCat].extend(myDict[cat])
            
        for newCat in newDict.keys():
            newDict[newCat] = frozenset(newDict[newCat])

        newLexicon = Lexicon()
        newLexicon.setLexicon(newDict)
        return newLexicon

    def unGroupCategories(self):
        newDict = {}
        myDict = self.currentLexicon

        for cat, terms in myDict.iteritems():
            for term in terms:
                newDict[term.replace(' ', '_')+"_"+cat] = [term]
            
        for newCat in newDict.keys():
            newDict[newCat] = frozenset(newDict[newCat])

        newLexicon = Lexicon()
        newLexicon.setLexicon(newDict)
        return newLexicon

    def union(self, otherLexicon):
        """union self lexicon with another and returns the result"""
        newDict = {}
        otherDict = otherLexicon.currentLexicon
        myDict = self.currentLexicon
        #print myDict
        #print otherDict
        unionKeys = set(myDict.keys()).union(set(otherDict.keys()))
        for cat in unionKeys:
            if (cat in myDict) and (cat in otherDict):
                newDict[cat] = frozenset(set(myDict[cat]).union(set(otherDict[cat])))
            else:
                if (cat in myDict):
                    newDict[cat] = frozenset(myDict[cat])
                else:
                    newDict[cat] = frozenset(otherDict[cat])
                
        newLexicon = Lexicon()
        newLexicon.setLexicon(newDict)
        return newLexicon

    def intersect(self, otherLexicon):
        """intersects self lexicon with another and returns the result"""
        newDict = {}
        otherDict = otherLexicon.currentLexicon
        myDict = self.currentLexicon
        unionKeys = set(myDict.keys()).union(set(otherDict.keys()))
        for cat in unionKeys:
            if (cat in myDict) and (cat in otherDict):
                newDict[cat] = frozenset(set(myDict[cat]).intersection(set(otherDict[cat])))
                
        newLexicon = Lexicon()
        newLexicon.setLexicon(newDict)
        return newLexicon
        

    def randomize(self):
        """randomizes the categories of the current lexicon"""
        myDict = self.currentLexicon
        myKeys = myDict.keys()
        newDict = {}
        for terms in self.currentLexicon.values():
            for term in terms:
                randKey = myKeys[random.randint(0,len(myKeys)-1)]
                if not(randKey in newDict):
                    newDict[randKey] = []
                newDict[randKey].append(term)
        for cat, terms in newDict.iteritems():
            newDict[cat] = frozenset(terms)
        newLex = Lexicon(newDict)
        newLex.setLexicon(newDict)
        return newLex


    def addTermsToCorpus(self, corpdb, corptable, termfield, messagefield, messageidfield, fulltext = False):
        """find rows with terms from lexicon and insert them back in as annotated rows"""
        #TODO: num_words and num_matches is hard-coded
        termList = set()
        for terms in self.currentLexicon.values():
            termList = termList.union(terms)
        termREs = {}
        #escapedTerm = re.escape(term).replace('\\*', '\w*') #handle punctuation / add wildcard
        termREs = dict((term, re.compile(r'\b(%s)\b' % re.escape(term).replace('\\*', '\w*'), re.I)) for term in termList)
        termLCs = dict((term, term.rstrip('*').lower()) for term in termList)

        (corpDb, corpCursor) = abstractDBConnect(HOST, USER, corpdb)
        writeCursor = corpDb.cursor()
        #(corpDb, writeCursor) = abstractDBConnect(HOST, USER, corpdb)
        #get field list:
        sql = """SELECT column_name FROM information_schema.columns WHERE table_name='%s' and table_schema='%s'""" % (corptable, corpdb)
        print sql
        corpCursor.execute(sql)
        rows = corpCursor.fetchall()
        fields = map(lambda r: r[0], rows)
        fieldIndex = dict((fields[i], i) for i in range(len(fields)))

        #Go through each message      
        try:
            sql = """SELECT * FROM %s GROUP BY %s""" % (corptable, messageidfield)
            print sql
            corpCursor.execute(sql)
        except MySQLdb.Error, e:
            warn("MYSQL ERROR in addTermsToCorpus:" + str(e) + sqlQuery);
            sys.exit(1)
        newRows = []
        row = corpCursor.fetchone()
        record = 0
        sqlQuery = """REPLACE INTO """+corptable+""" values ("""+""", """.join(list('%s' for f in fields))+""")"""
        while row:
            if (len(row) != len(fields)):
                warn("row not correct size: " + str(row))
                sys.exit(1)

            else:
                record+=1
                message = row[fieldIndex[messagefield]]

                #first set number of words
                num_words = 0
                if message: 
                    num_words = len(message.split())

                message_id = row[fieldIndex[messageidfield]]
                updateSql = """UPDATE %s set num_words = %s WHERE %s = '%s'""" % (corptable, str(num_words), messageidfield, str(message_id))
                try:
                        #newRows = map (lambda r: r[f] = r[f].replace("'",  for f in range(fields))
                    writeCursor.execute(updateSql)
                except Exception, e:
                    warn("Exception during mysql call:" + str(e) + ': ' + sqlQuery);
                    pprint.PrettyPrinter().pprint(newRows) #debug
                    sys.exit(1)
            
                #then add extra rows:
                for term in termList:
                    if message and termLCs[term] in message.lower():
                        num_matches = len(termREs[term].findall(message))
                        if num_matches > 0:
                            newRow = list(row)
                            newRow[fieldIndex[termfield]] = term
                            newRow[fieldIndex['num_words']] = num_words
                            newRow[fieldIndex['num_matches']] = num_matches
                            newRow[fieldIndex['id']] = str(row[fieldIndex[messageidfield]])+'.'+term
                            newRows.append(newRow)
                if record % MAX_WRITE_RECORDS == 0: 
                    print "\n writing new rows up to: %d " % record
                    #write them back in:
                    try:
                        #newRows = map (lambda r: r[f] = r[f].replace("'",  for f in range(fields))
                        writeCursor.executemany(sqlQuery, newRows)
                        newRows = []
                    except Exception, e:
                        warn("Exception during mysql call:" + str(e) + ': ' + sqlQuery);
                        pprint.PrettyPrinter().pprint(newRows) #debug
                        sys.exit(1)

            row = corpCursor.fetchone()

        print "Writing remaining. %d total rows checked" % record
        try:
            writeCursor.executemany(sqlQuery, newRows)
        except Exception, e:
            warn("Exception during mysql call:" + str(e) + ': ' + sqlQuery);
            sys.exit(1)


    def createLexiconFromCorpus(self, corpdb, corptable, messagefield, messageidfield, minwordfreq):
        """Creates a lexicon (all in one category) from a examining word frequencies in a corpus"""
        wordList = dict()
        
        (corpDb, corpCursor) = abstractDBConnect(HOST, USER, corpdb)

        #Go through each message      
        try:
            sql = """SELECT %s FROM %s""" % (messagefield, corptable)
            if messageidfield: 
                sql += """ GROUP BY %s""" %  messageidfield
            warn(sql+"\n")
            corpCursor.execute(sql)
        except MySQLdb.Error, e:
            warn("MYSQL ERROR1:" + str(e) + sqlQuery)
            sys.exit(1)
        row = corpCursor.fetchone()
        rowNum = 0
        while row:
            rowNum+=1
            if rowNum % 10000 == 0:
                warn("On Row: %s\n" % rowNum)
            message = row[0]
            words = message.split()
            for word in words:
                if not word in wordList:
                    wordList[word] = 0
                wordList[word] += 1
            row = corpCursor.fetchone()

        #now create a set based on the words that occur more than minwordfreq
        lex = set()
        for word, freq in wordList.iteritems():
            if (freq >= minwordfreq):
                lex.add(word)
        
        self.currentLexicon = {'word': lex}

    def annotateSenses(self, currentName, newLexiconName):
        #prompts the user to annotate the current lexicon, continues annotating if it exists
        weighted = self.isTableLexiconWeighted(currentName)
        senseLexicon = WeightedLexicon()
        try: 
            senseLexicon.loadLexicon(newLexiconName)
        except MySQLdb.Error, e:
            print "in except"
            #couldn't load lexicon, create it but empty
            createLike = """CREATE TABLE %s LIKE %s""" % (newLexiconName, currentName)
            self.dbCursor.execute(createLike)
            senseLexicon.loadLexicon(newLexiconName)

        #get lexicon dicts:
        oldLexicon = self.currentLexicon
        newLexicon = senseLexicon.currentLexicon

        #find what's been done
        seenWords = set()
        for cat, words in newLexicon.iteritems():
            cat = cat.lower()
            for ws in words:
                if ws:
                    (lemma, pos, sense, word) = ws.split('.')
                    seenWords.add(cat+'#'+word)

        #prompt for new words
        for cat, words in oldLexicon.iteritems():
            cat = cat.lower()
            for word in words:
                if cat+'#'+word in seenWords:
                    print "already annotated %s: %s (skipping)" % (cat, word)
                else:
                    senses = interactiveGetSenses(cat, word)
                    if senses:
                        if weighted:
                            smallNewLex = {cat: dict(zip(senses, [words[word]]*len(senses)))}
                            sys.stderr.write("newLexiconName %s \n" % newLexiconName)
                            self.insertWeightedLexiconRows(newLexiconName, smallNewLex)
                        else: 
                            smallNewLex = {cat: frozenset(senses)}
                            self.insertLexiconRows(newLexiconName, smallNewLex)
                    
    def expand(self):
        """Expands a lexicon to contain more words"""
        newLexicon = dict()
        for cat, words in self.currentLexicon.iteritems():
            newLexicon[cat] = set()
            for word in words:
                print word #debug
                otherWords = Lexicon.wordExpand(word)
                newLexicon[cat] = newLexicon[cat].union(otherWords)
        
        return Lexicon(newLexicon)

    wpsRE = re.compile(r'[nvar]\.(\d+|\?)$')
    wpRE = re.compile(r'^([^\.#]+)\.([a-z])', re.I)

    @staticmethod
    def wordExpand(word,  specificDepth = 1, generalizeDepth = -1, totalLinks = 2):
        #specificDepth = 2 #length to travel toward more specific words
        #generalizeDepth = -1 #length (negative) to travel toward more general words
        # totalLinks: max links to travel along
        word = word.replace('#', '.')
        wpss = []
        words = set()
        if not Lexicon.wpsRE.search(word):
            match = Lexicon.wpRE.match(word)
            if match:
                pos = match.group(2)
                wpss = [(w, 0, 0) for w in wn.synsets(match.group(1)+'.'+pos)]
                words.add(match.group(1))
            else:
                wpss = [(w, 0, 0) for w in wn.synsets(word)]
                words.add(word)
        else:
            wpss = [(wn.synset(word), 0, 0)]

        seen = set()
        while wpss:
            wps, depth, links = wpss.pop()
            if wps not in seen:

                #add word to list:
                lemmas = [wps]
                if not isinstance(wps, str):
                    lemmas = [w.name for w in wps.lemmas]
                for lemma in lemmas:
                    wordMatch = Lexicon.wpRE.match(lemma)
                    if wordMatch:
                        words.add(wordMatch.group(1))
                    else:
                        words.add(lemma)

                #find neighbors:
                if links < totalLinks:
                    if depth < specificDepth:
                        #add hyponyms:
                        hypos = set(wps.hyponyms()  + 
                                     wps.entailments())
                        for hypo in hypos:
                            #print "%s hypo: %s" % (wps, hypo)
                            wpss.append((hypo, depth+1, links+1))
                    if depth > generalizeDepth:
                        #add hypernyms:
                        hypers = set(wps.hypernyms() + wps.instance_hypernyms() +
                                    wps.verb_groups())
                        for hype in hypers:
                            #print "%s hyper: %s" % (wps, hype)
                            wpss.append((hype, depth-1, links+1))

                seen.add(wps)

        #print "%s: %s" % (word, str(words))
        words = set([w.replace('_', ' ') for w in words])
        return words

    def likeExamples(self, corpdb, corptable, messagefield, numForEach = 60, onlyPrintIfMin = True, onlyPrintStartingAlpha = True):
        (corpDb, corpCursor) = abstractDBConnect(HOST, USER, corpdb)

        print "<html><head>"
        print "<style>"
        print "br {line-height: 6pt;}"
        print "li {margin:0; padding:0; margin-bottom:10pt;}"
        print "</style></head><body><table>"


        #csvFile = open('/tmp/examples.csv', 'w')

        for cat, terms in self.currentLexicon.iteritems():
            print "<tr><td colspan=3> </td></tr>"
            print "<tr><td colspan=3><h3>%s</h3></td></tr>" % (cat)

            for term in sorted(terms):
                if term[0].isalpha() or not onlyPrintStartingAlpha:
                    escTerm = re.escape(term).replace('\\*', '\w*').replace('\\ ', '[[:space:]]').replace('(', '\(').replace(')', '\)') #handle punctuation / add wildcard
                    sql = """SELECT %s FROM %s WHERE %s RLIKE '[[:<:]]%s[[:>:]]'""" % (messagefield, corptable, messagefield, escTerm);
                    #print sql #debug
                    corpCursor.execute(sql)
                    messages = [x[0] for x in corpCursor.fetchall()]
                    lenmsgs = len(messages)
                    if not onlyPrintIfMin or lenmsgs >= numForEach:
                        print "<tr><td><b>%s</b></td><td><b>%s</b></td><td><em>%d occurrences</em></td></tr>" % (cat, term, lenmsgs)
                        print "<tr><td><b>%s</b></td><td><b>%s</b></td><td><b>%s</b></td></tr>" % ("Correct?", "term", "comment")

                        toPrint = [] #holds messages to print
                        if lenmsgs < numForEach:
                            toPrint = messages
                        else:
                            done = set()
                            while len(toPrint) < numForEach:
                                i = random.randint(0, lenmsgs -1)
                                if i not in done:
                                    toPrint.append(messages[i])
                                    done.add(i)

                        tre = re.compile("\\b%s\\b"%(term), re.I)
                        for m in toPrint:
                            print "<tr><td></td><td>%s</td><td>%s</td></tr>" % (term, tre.sub("<b>%s</b>"%term, m))
                        print "<tr><td colspan=3> </td></tr>"
        
        print "</table></body></html>"

    def likeSamples(self, corpdb, corptable, messagefield, category, lexicon_name, number_of_messages):
        (corpDb, corpCursor) = abstractDBConnect(HOST, USER, corpdb)

        #csvFile = open('/tmp/examples.csv', 'w')
        csvFile = open(lexicon_name+"_"+category+'.csv','wb')
#        csvFile.write('"id", "message", "term"\n')
        messages = list()
        def findTerm((m_id, string, term)):
            if 'space' in term:
                term = re.sub(r"\[\[:space:\]\]",r"\s",term, re.I)
            match = re.findall(term, string, re.I)
            return (m_id,string,match[0])
        for term in sorted(self.currentLexicon[category]):
            escTerm = re.escape(term).replace('\\*', '\w*').replace('\\ ', '[[:space:]]+') #handle punctuation / add wildcard
            sql = """SELECT id, %s FROM %s WHERE %s RLIKE '[[:<:]]%s[[:>:]]'""" % (messagefield, corptable, messagefield, escTerm);
                #print sql #debug
            corpCursor.execute(sql)
            print "Looking for messages containing %s" %  escTerm
            new_ones = [(x[0],x[1],escTerm) for x in corpCursor.fetchall()]
            new_ones = map(findTerm,new_ones)
            messages.extend(new_ones)
        
        writer=csv.writer(csvFile,dialect='excel')
        random.shuffle(messages)
        writer.writerows(messages[:number_of_messages])
        

    def printCSV(self):
        """prints a csv style output of the lexicon"""
        ##print headers:
        print ','.join(self.currentLexicon.iterkeys())

        ##turn sets into lists
        lexList = {}
        longest = 0;
        for cat, terms in self.currentLexicon.iteritems():
            lexList[cat] = list(terms)
            longest = max(longest, len(terms))
        
        for i in range(longest):
            row = []
            for cat in self.currentLexicon.iterkeys():
                if (i < len(lexList[cat])):
                    row.append(lexList[cat][i])
                else:
                    row.append('')
            print ','.join(row)

    # def printWeightedCSV(self):
    #     """prints a csv style output of the lexicon"""
    #     ##print headers:
    #     for cat, values in self.weightedLexicon.iteritems():
            
    #         topWords = sorted(values.items(), key=lambda x: x[1], reverse=True)[:30]
    #         topWString = ",",join(["'"+k+"',"+str(v) for k,v in topWords])
    #         print cat+','+topWString

            
    def pprint(self):
        """Uses pprint to print the current lexicon"""
        if self.currentLexicon:
            pprint.PrettyPrinter().pprint(self.currentLexicon)
        else:
            pprint.PrettyPrinter().pprint(self.weightedLexicon)

    def compare(self, otherLex):
        """Compares two lexicons, depends on the types"""
        uLexs = []
        wLexs = []
        for lex in (self, otherLex):
            if 'weightedLexicon' in lex.__dict__ and lex.weightedLexicon:
                wLexs.append(lex)
            else:
                uLexs.append(lex)

        if len(wLexs) > 1:
            return WeightedLexicon.compareWeightedToWeighted(self.weightedLexicon, otherLex.weightedLexicon)
        elif (len(wLexs) == 1):
            return WeightedLexicon.compareWeightedToUnweighted(wLexs[0].weightedLexicon, uLexs[0].currentLexicon)
        else:
            return WeightedLexicon.compareUnweightedToUnweighted(self.currentLexicon, otherLex.currentLexicon)


   

class WeightedLexicon(Lexicon):
    """WeightedLexicons have an additional dictionary with weights for each term in the regular lexicon"""
    def __init__(self, weightedLexicon=None, lex=None, mysql_host = HOST):
        super(WeightedLexicon, self).__init__(lex, mysql_host = mysql_host)
        print self.mysql_host
        self.weightedLexicon = weightedLexicon
     
    def isTableLexiconWeighted(self, tablename):
        sql = "SHOW COLUMNS from %s"%tablename
#        try:
        self.dbCursor.execute(sql)
        data = self.dbCursor.fetchall()
#        except MySQLdb.Error, e:
#            warn(" MYSQL ERROR" + str(e))
#            sys.exit(1)
        if len(data) > 0:
            numColumns = len(data)
            if numColumns == 3:
                return False
            elif numColumns == 4:
                return True
            else:
                raise Exception("Incorrect lexicon specified; number of rows in table [%s] is not 3 or 4")
        else:
            raise Exception("Lexicon table [%s] has no columns"%tablename)
            
        
    def isSelfLexiconWeighted(self):
        if self.weightedLexicon and isinstance(self.weightedLexicon, dict):
            for words in self.weightedLexicon.itervalues():
                if isinstance(words, dict):
                    return True
                else:
                    return False
        else:
            return False
            
    def loadLexicon(self, tablename, where=''):
        """Loads a lexicon, checking to see if it is weighted or not, then responding accordingly"""
        if self.isTableLexiconWeighted(tablename):
            return self.loadWeightedLexicon(tablename, where)
        else:
            return super(WeightedLexicon, self).loadLexicon(tablename, where)
  
    def loadWeightedLexicon(self, tablename, where = ''):
        """Loads a lexicon as weightedLexicon"""
        sqlQuery = "SELECT term, category, weight from %s"%tablename
        if where:
            sqlQuery += ' WHERE ' + where
        
        data = []
#        try:
        self.dbCursor.execute(sqlQuery)
        data = self.dbCursor.fetchall()
#        except MySQLdb.Error, e:
#            warn(" MYSQL ERROR" + str(e))
#            sys.exit(1)
        lexicon = {}
        for (term, category, weight) in data:
            if not category in lexicon:
                lexicon[category] = {}
            lexicon[category][term] = weight

        self.weightedLexicon = lexicon
        return self.weightedLexicon
     
    def getWeightedLexicon(self):
        return self.weightedLexicon

    def createLexiconTable(self, tablename):
        """Loads a lexicon, checking to see if it is weighted or not, then responding accordingly"""
        if self.isSelfLexiconWeighted():
            print "Creating weighted lexicon table"
            return self.createWeightedLexiconTable(tablename)
        else:
            print "Creating unweighted lexicon table"
            return super(WeightedLexicon, self).createLexiconTable(tablename)

    def createWeightedLexiconTable(self, tablename):
        """Creates a lexicon table from the instance's lexicon variable"""
        
        #first create the table:
        enumCats = "'"+"', '".join(map(lambda k: k.upper().replace("'", "\\'"), self.weightedLexicon.keys()))+"'"   
        drop = """DROP TABLE IF EXISTS """+tablename
        sql = """CREATE TABLE IF NOT EXISTS %s (id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, term VARCHAR(140), category ENUM(%s), weight DOUBLE, INDEX(term), INDEX(category)) ENGINE = MyISAM""" % (tablename, enumCats)
        print "Running: ", drop
        print "and:     ", sql
        try:
            self.dbCursor.execute(drop)
            self.dbCursor.execute(sql)
        except MySQLdb.Error, e:
            warn("MYSQL ERROR2" + str(e))
            sys.exit(1)

        #next insert rows:
        self.insertWeightedLexiconRows(tablename)
        print "Done creating lexicon: %s" % tablename

    def insertWeightedLexiconRows(self, tablename, lex = None):
        """Adds rows, taken from the lexicon variable to mysql"""
        if not lex: lex = self.weightedLexicon
        sqlQuery = """INSERT INTO """+tablename+""" (term, category, weight) values (%s, %s, %s)"""
        values = []
        for cat in lex:
            for term in lex[cat]:
                # print 'cat: %s term: %s' % (cat, term)
                if self.weightedLexicon[cat][term] != 0:
                    values.extend([[term, cat, self.weightedLexicon[cat][term]]])
        try:
            nbInserted = 0
            length = len(values)
            for v in zip(*[iter(values)]*100):
                nbInserted += self.dbCursor.executemany(sqlQuery, v)    
            remainingValues = values[nbInserted:]
            if remainingValues:
                nbInserted += self.dbCursor.executemany(sqlQuery, remainingValues)
            print "Inserted %d terms into the lexicon" % nbInserted
            if nbInserted != length:
                print "Warning the number of rows inserted doesn't match the total number of rows"
                

        except MySQLdb.Error, e:
            warn("MYSQL ERROR:" + str(e) + sqlQuery);
            sys.exit(1)

    def mapToSuperLexicon(self, superLexiconMapping):
        """Creates a new lexicon based on mapping topic words to super topics """
        lex = self.weightedLexicon #category->word->weight ; lex[cat][word]
        mapping = superLexiconMapping.weightedLexicon
        superTopicDict = dict() #[word] -> [super_topic]-> [combined_weight]
        for superTopic, topicDict in mapping.iteritems(): #super-topics
            superTopicDict[superTopic] = dict()
            for topic, topicWeight in topicDict.iteritems():
                if topic in lex:
                    for word, wordWeight in lex[topic].iteritems():
                        try: 
                            superTopicDict[superTopic][word] += wordWeight*topicWeight
                        except KeyError:
                            superTopicDict[superTopic][word] = wordWeight*topicWeight
                else:
                    print "Warning topic %s in super-topic but not original topic lexicon" % str(topic)

        return WeightedLexicon(weightedLexicon=superTopicDict)
            
    def union(self, otherLexicon):
        """union self lexicon with another and returns the result"""
        if not self.isSelfLexiconWeighted():
            print "union: not weighted"
            return super(WeightedLexicon, self).union(otherLexicon)
        newDict = {}
        otherDict = otherLexicon.weightedLexicon
        myDict = self.weightedLexicon
        unionKeys = set(myDict.keys()).union(set(otherDict.keys()))
        for cat in unionKeys:
            if (cat in myDict) and (cat in otherDict):
                newDict[cat] = dict(myDict.items() + otherDict.items())
            else:
                if (cat in myDict):
                    newDict[cat] = myDict[cat]
                else:
                    newDict[cat] = otherDict[cat]
                
        newLexicon = WeightedLexicon(newDict)
        return newLexicon

    def compare(self, otherLex):
        """Compares two lexicons, depends on the types"""
        uLexs = []
        wLexs = []
        for lex in (self, otherLex):
            if 'weightedLexicon' in lex.__dict__ and lex.weightedLexicon:
                wLexs.append(lex)
            else:
                uLexs.append(lex)

        if len(wLexs) > 1:
            return WeightedLexicon.compareWeightedToWeighted(self.weightedLexicon, otherLex.weightedLexicon)
        elif (len(wLexs) == 1):
            return WeightedLexicon.compareWeightedToUnweighted(wLexs[0].weightedLexicon, uLexs[0].currentLexicon)
        else:
            return WeightedLexicon.compareUnweightedToUnweighted(self.currentLexicon, otherLex.currentLexicon)


    def printCSV(self):
        """prints a csv style output of the lexicon"""
        if not self.isSelfLexiconWeighted():
            super(WeightedLexicon, self).printCSV()

        else:
            ##print headers:
            print "category, term1, w1, term2, w2, ..."

            ##print categories
            for cat, terms in self.weightedLexicon.iteritems():
                print cat+','+','.join(["%s,%d"%(term, w) for term, w in sorted(terms.iteritems(), key = lambda x: x[1], reverse=True)[:20]])

    def lextagcloud(self, outputname=None):
        """prints a tagcloud style output of the lexicon"""
        nterms = 15
        minsize = 10
        maxbrightness = .4 # of each word, where 0=black and 1=white

        if not outputname[-4:]=='.txt':
            outputname += '.txt'


        ##print categories
        with open(outputname,'w') as f:
            for cat, terms in self.weightedLexicon.iteritems():
                f.write('[Topic Id: %s]\n' % cat)
                for i, termw in enumerate(sorted(terms.iteritems(), key = lambda x: x[1], reverse=True)[:nterms]):
                    term, w = termw
                    termsize = minsize+nterms-i
                    termcolor = hex(int(255*random.uniform(0,maxbrightness)))[-2:]*3
                    f.write('%s:%d:%s\n' % (term, termsize, termcolor))
                f.write('\n\n')

                #print cat+','+','.join(["%s,%d"%(term, w) for term, w in sorted(terms.iteritems(), key = lambda x: x[1], reverse=True)[:20]])
            

    @staticmethod
    def compareWeightedToWeighted(wLex1, wLex2):
        """Compares two weighted lexicons"""
        #both wLex1 and wLex2 are dict of dicts: [category][word] = weight
        #TODO: Achal
        print "comparing weighted to weighted" #debug
        similarity = dict() #[cat1][cat2] = similarity_value
        for cat1 in wLex1.iterkeys():
            similarity[cat1] = dict()
            for cat2 in wLex2.iterkeys():
                similarity[cat1][cat2] = 0 #TODO

        return similarity

    @staticmethod
    def compareWeightedToUnweighted(wLex1, uLex2, comparisonMethod='weighted'):
        """Compares a weighted lexicon (self) to an unweighted lexicon"""
        #both wLex1 is a dict of dicts: [category][word] = weight
        #uLex2 is a dict of sets: [category]=set(<terms in category>)
        #TODO: Achal
        # shoudl either convert uLex2 to wLex2 and run compareWeighted.. or
        # convert wLex1 to uLex1 (by thresholding) and run compareUnweighted
        #(may not even use this code below)
        print "comparing weighted to unweighted" #debug
        similarity = dict() #[cat1][cat2] = similarity_value
        for cat1 in wLex1.iterkeys():
            similarity[cat1] = dict()
            for cat2 in uLex2.iterkeys():
                similarity[cat1][cat2] = 0 #TODO

        return similarity

    @staticmethod
    def compareUnweightedToUnweighted(uLex1, uLex2, metric='jaccard'):
        """Compares two unweighted lexicons"""
        #both uLex1 and uLex2 are dict of sets
        #TODO: Achal
        #run either jaccard or cosine sim (with probs equal to 1)
        print "comparing unweighted to unweighted" #debug
        similarity = dict() #[cat1][cat2] = similarity_value
        for cat1 in uLex1.iterkeys():
            similarity[cat1] = dict()
            for cat2 in uLex2.iterkeys():
                similarity[cat1][cat2] = 0 #TODO

        return similarity


###########################################################
## Command-prompt helper functions:

###########################################################
## Main / Command-prompt Interface Area
#

if __name__ == "__main__":
##SETUP ARGUMENTS##
    _optParser = OptionParser()

    _optParser.add_option("-f", "--file", dest="filename",
                          help="Lexicon Filename")
    _optParser.add_option("-g", "--gfile", dest="gfile",
                          help="Lexicon Filename in google format")
    _optParser.add_option("--sparsefile", dest="sparsefile",
                          help="Lexicon Filename in sparse format")
    _optParser.add_option("--weightedsparsefile", dest="weightedsparsefile",
                          help="Lexicon Filename in weighted sparse format")
    _optParser.add_option("--dicfile", dest="dicfile",
                          help="Lexicon Filename in dic (LIWC) format")
    _optParser.add_option("--topicfile", dest="topicfile",
                          help="Lexicon Filename in topic format")
    _optParser.add_option("--topic_csv", "--weighted_file", action='store_true', dest="topiccsv", default=False,
                          help="tells interface to use the topic csv format to make a weighted lexicon")
    _optParser.add_option("--filter", action="store_true", dest="using_filter",
                          help="Allows lexicon filtering if True")
    _optParser.add_option("-n", "--name", dest="name",
                          help="Existing Lexicon Table Name (will load)")
    _optParser.add_option("-c", "--create", dest="create",
                          help="Create a new lexicon table (must supply new lexicon name, and either -f, -g or -n)")
    _optParser.add_option("-p", "--print", action="store_true", dest="printcsv",
                          help="print lexicon to stdout (default csv format)")
    _optParser.add_option("--print_weighted", action="store_true", dest="printweightedcsv",
                          help="print lexicon to stdout (weighted csv format)")
    _optParser.add_option("--lex_tagcloud", action="store_true", dest="lextagcloud",
                          help="print lexicon tagcloud text files to --output_name")
    _optParser.add_option("--output_name", dest="outputname",
                          help="path to output lexicon tagclouds")
    _optParser.add_option("--pprint", action="store_true", dest="pprint",
                          help="print lexicon to stdout as pprint output")
    _optParser.add_option("-w", "--where", dest="where",
                          help="where phrase to add to sql query")
    _optParser.add_option("-u", "--union", dest="union",
                          help="Unions two tables and uses the result as myLexicon")
    _optParser.add_option("-i", "--intersect", dest="intersect",
                          help="Intersects two tables and uses the result as myLexicon")
    _optParser.add_option("--super_topic", type=str, dest="supertopic",
                          help="Maps the current lexicon with a super topic mapping lexicon to make a super_topic"),
    _optParser.add_option("-r", "--randomize", action="store_true", dest="randomize",
                          help="Randomizes the categories of terms")
    _optParser.add_option("--depol", action="store_true", dest="depol",
                          help="Depolarize the categories (removes +/-)")
    _optParser.add_option("--ungroup", action="store_true", dest="ungroup", default = False,
                          help="places each word in its own category")
    _optParser.add_option("--compare", dest="compare",
                          help="Unions two tables and uses the result as myLexicon")
    _optParser.add_option("--annotate_senses", dest="sense_annotated_lex", type=str,
                          help="Asks the user to annotate senses of words and creates a new lexicon with senses (new lexicon name is the parameter)")

    _optParser.add_option("--topic_threshold", type=float, dest="topicthreshold", default=None,
                          help="sets the threshold to use for a csv topicfile")

    _optParser.add_option("-a", "--add_terms", action="store_true", dest="addterms",
                          help="Adds terms from the loaded lexicon to a given corpus (options below)")
    _optParser.add_option("-l", "--corpus_lexicon", action="store_true", dest="corpuslex",
                          help="Load a lexicon based on finding words in a given corpus (BETA) (options below)")
    _optParser.add_option("--corpus_examples", action="store_true", dest="examples",
                          help="Find example instances of words in the given corpus (using rlike; equal number for all words)")
    _optParser.add_option("--corpus_samples", action="store_true", dest="samples",
                          help="Find sample of matches for lexicon.")
    _optParser.add_option("-e", "--expand_lexicon", action="store_true", dest="expand",
                          help="Expands the lexicon to more terms.")

    group = OptionGroup(_optParser, "Add Terms OR Corpus Lexicon Options","")
    group.add_option("-d", "--corpus_db", dest="corpdb", metavar='DB', default = DEF_CORPDB,
                         help="Corpus database to use [default: %default]")
    group.add_option("-t", "--corpus_table", dest="corptable", metavar='TABLE', default = DEF_CORPTABLE,
                         help="Corpus table to use [default: %default]")
    group.add_option("--corpus_term_field", dest="termfield", metavar='FIELD', default = DEF_TERMFIELD    ,
                         help="field of the corpus table that contains terms (lexicon table always uses 'term') [default: %default]")
    group.add_option("--corpus_message_field", dest="messagefield", metavar='FIELD', default = DEF_MESSAGEFIELD    ,
                         help="field of the corpus table that contains the actual message [default: %default]")
    group.add_option("--corpus_messageid_field", dest="messageidfield", metavar='FIELD', default = DEF_MESSAGEIDFIELD    ,
                         help="field of the table that contains message ids (set to '' to not use group by [default: %default]")
    group.add_option("--min_word_freq", dest="minwordfreq", metavar='NUM', type='int', default = DEF_MINWORDFREQ    ,
                         help="minimum number of instances to include in lexicon (-l option) [default: %default]")
    group.add_option("--lexicon_category", dest="lexicon_cat", metavar="CATEGORY", 
                         help="category in lexicon to get random samples from")
    group.add_option("--num_rand_messages", dest="num_messages", metavar="NUM", type='int', default = DEF_NUMRANDMESSAGES,
                         help="number of random messages to select when getting samples from lexicon category")
#    group.add_option("--fulltext", action="store_true", dest="fulltext", default = False,
#                          help="utilizes fulltext searches to improve performance (TODO)")


    _optParser.add_option_group(group)


    (_options,_args) = _optParser.parse_args()
    
##DETERMINE WHAT FUNCTION TO RUN##
    myLexicon = None
    if _options.name:
        myLexicon = WeightedLexicon()
        myLexicon.loadLexicon(_options.name, _options.where)
    if _options.filename:
        myLexicon = WeightedLexicon()
        myLexicon.setLexicon(loadLexiconFromFile(_options.filename))
    if _options.gfile:
        myLexicon = WeightedLexicon()
        myLexicon.setLexicon(loadLexiconFromGFile(_options.gfile, _options.using_filter))
    if _options.sparsefile:
        myLexicon = WeightedLexicon()
        myLexicon.setLexicon(loadLexiconFromSparse(_options.sparsefile))
    if _options.dicfile:
        myLexicon = WeightedLexicon()
        myLexicon.setLexicon(loadLexiconFromDic(_options.dicfile))
    if _options.weightedsparsefile:
        myLexicon = WeightedLexicon(loadWeightedLexiconFromSparse(_options.weightedsparsefile))
    if _options.topicfile:
        myLexicon = WeightedLexicon()
        if _options.topiccsv:
            myLexicon = WeightedLexicon(loadWeightedLexiconFromTopicCSV(_options.topicfile, _options.topicthreshold))
        else:
            myLexicon.setLexicon(loadLexiconFromTopicFile(_options.topicfile))
    if _options.corpuslex:
        myLexicon = WeightedLexicon()
        myLexicon.createLexiconFromCorpus(_options.corpdb, _options.corptable, _options.messagefield, _options.messageidfield, _options.minwordfreq)
    if _options.union:
        if not myLexicon:
            print "Must load a lexicon, either from a file (-f), or from another table (-n)"
            sys.exit()
        otherLexicon = WeightedLexicon()
        otherLexicon.loadLexicon(_options.union)
        myLexicon = myLexicon.union(otherLexicon)
    if _options.intersect:
        if not myLexicon:
            print "Must load a lexicon, either from a file (-f), or from another table (-n)"
            sys.exit()
        otherLexicon = WeightedLexicon()
        otherLexicon.loadLexicon(_options.intersect)
        myLexicon = myLexicon.intersect(otherLexicon)
    if _options.supertopic:
        superLexiconMapping = WeightedLexicon()
        superLexiconMapping.loadLexicon(_options.supertopic)
        myLexicon = myLexicon.mapToSuperLexicon(superLexiconMapping)
    if _options.randomize:
        if not myLexicon:
            print "Must load a lexicon, either from a file (-f), or from another table (-n)"
            sys.exit()
        myLexicon = myLexicon.randomize()
    if _options.depol:
        if not myLexicon:
            print "Must load a lexicon, either from a file (-f), or from another table (-n)"
            sys.exit()
        myLexicon = myLexicon.depolCategories()
    if _options.ungroup:
        if not myLexicon:
            print "Must load a lexicon, either from a file (-f), or from another table (-n)"
            sys.exit()
        myLexicon = myLexicon.unGroupCategories()
    if _options.compare:
        if not myLexicon:
            print "Must load a lexicon, either from a file (-f), or from another table (-n)"
            sys.exit()
        otherLexicon = WeightedLexicon()
        otherLexicon.loadLexicon(_options.compare)
        pprint.PrettyPrinter().pprint(myLexicon.compare(otherLexicon))

    if _options.sense_annotated_lex:
        if not myLexicon:
            print "Must load a lexicon, either from a file (-f), or from another table (-n)"
            sys.exit()
        myLexicon.annotateSenses(_options.name, _options.sense_annotated_lex)

    if _options.expand:
        if not myLexicon:
            print "Must load a lexicon, either from a file (-f), or from another table (-n)"
            sys.exit()
        myLexicon = myLexicon.expand()

    if _options.create:
        if not myLexicon:
            print "Must load a lexicon, either from a file (-f), or from another table (-n)"
            sys.exit()
        myLexicon.createLexiconTable(_options.create)
    if _options.addterms:
        if not myLexicon:
            print "Must load a lexicon, either from a file (-f), or from another table (-n)"
            sys.exit()
        myLexicon.addTermsToCorpus(_options.corpdb, _options.corptable, _options.termfield, _options.messagefield, _options.messageidfield, _options.fulltext)
    if _options.examples:
        if not myLexicon:
            print "Must load a lexicon, either from a file (-f), or from another table (-n)"
            sys.exit()
        myLexicon.likeExamples(_options.corpdb, _options.corptable, _options.messagefield)
    if _options.samples:
        if not _options.lexicon_cat:
            print "Must specify a lexicon category with option '--lexicon_cat'"
            sys.exit()
        myLexicon.likeSamples(_options.corpdb, _options.corptable, _options.messagefield, _options.lexicon_cat, _options.name, _options.num_messages)
    if _options.printcsv:
        if not myLexicon:
            print "Must load a lexicon, either from a file (-f), or from another table (-n)"
            sys.exit()
        myLexicon.printCSV()
    if _options.lextagcloud:
        if not myLexicon or not _options.outputname:
            print "Must load a lexicon, either from a file (-f), or from another table (-n), and specify --output_name"
            sys.exit()
        myLexicon.lextagcloud(outputname=_options.outputname)
    if _options.printweightedcsv:
        if not myLexicon:
            print "Must load a lexicon, either from a file (-f), or from another table (-n)"
            sys.exit()
        myLexicon.printWeightedCSV()
    if _options.pprint:
        if not myLexicon:
            print "Must load a lexicon, either from a file (-f), or from another table (-n)"
            sys.exit()
        myLexicon.pprint()

