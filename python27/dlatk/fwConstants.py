import sys, os, getpass
import re
from random import randint

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

#math / stats:
from math import floor, log
from numpy import sqrt, log2, array, mean, std, isnan, fabs, round
from numpy.random import permutation
import numpy as np
from scipy.stats import zscore
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import statsmodels.stats.multitest as mt

###########################################################
### Constants
##
#

#DB INFO:
HOST = ''
USER = getpass.getuser()

MAX_ATTEMPTS = 5 #max number of times to try a query before exiting
PROGRESS_AFTER_ROWS = 5000 #the number of rows to process between each progress updated
FEATURE_TABLE_PREFIX = 'feats_'
MYSQL_ERROR_SLEEP = 4 #number of seconds to wait before trying a query again (incase there was a server restart
MYSQL_BATCH_INSERT_SIZE = 10000 # how many rows are inserted into mysql at a time
MYSQL_HOST = '127.0.0.1'
VARCHAR_WORD_LENGTH = 36 #length to allocate var chars per words
LOWERCASE_ONLY = 1#if the db is case insensitive, set to 1
MAX_TO_DISABLE_KEYS = 50000 #number of groups * n must be less than this to disable keys
MAX_SQL_PRINT_CHARS = 256

##Corpus Settings:
DEF_CORPDB = 'fb20'
DEF_CORPTABLE = 'messages_en'
DEF_CORREL_FIELD = 'user_id'
DEF_MESSAGE_FIELD = 'message'
DEF_MESSAGEID_FIELD = 'message_id'
DEF_ENCODING = 'utf8mb4'
DEF_UNICODE_SWITCH = True
DEF_LEXTABLE = 'wn_O'
DEF_DATE_FIELD = 'updated_time'
DEF_COLLATIONS = {
        'utf8mb4': 'utf8mb4_bin',
        'utf8': 'utf8_general_ci', 
        'latin1': 'latin1_swedish_ci', 
        'latin2': 'latin2_general_ci', 
        'ascii': 'ascii_general_ci',
    }

##Outcome settings
DEF_OUTCOME_TABLE = 'masterstats_andy'
DEF_OUTCOME_FIELD = 'SWL'
DEF_OUTCOME_FIELDS = []
DEF_OUTCOME_CONTROLS = []
DEF_GROUP_FREQ_THRESHOLD = int(1000) #min. number of total feature values that the group has, to use it
DEF_SHOW_FEAT_FREQS = True
DEF_MAX_TC_WORDS = 100
DEF_TC_FILTER = True

##Feature Settings:
DEF_N = int(1)
DEF_FEAT_NAMES = ['honor']
DEF_MIN_FREQ = int(1) #min frequency per group to keep (don't advise above 1)
DEF_P_OCC = float(.01) #percentage of groups a feature must appear in, to keep it
DEF_PMI = 3.0
DEF_MIN_FEAT_SUM = 0 #minimum sum of feature total to keep
DEF_STANFORD_POS_MODEL = '../Tools/StanfordTagger/stanford-postagger-2012-01-06/models/english-bidirectional-distsim.tagger'
DEF_LEXICON_DB = 'permaLexicon'
DEF_FEAT_TABLE = 'feat$1gram$messages_en$user_id$16to16$0_01'
DEF_COLLOCTABLE = 'test_collocs'
DEF_COLUMN_COLLOC = "feat"
DEF_COLUMN_PMI_FILTER = "pmi_filter_val"
DEF_P = 0.05 # p value for printing tagclouds
DEF_P_CORR = 'BH' #Benjamini, Hochberg
DEF_P_MAPPING = { # maps old R method names to statsmodel names
        "holm": "holm",  
        "hochberg": "simes-hochberg", 
        "simes": "simes", 
        "hommel": "hommel", 
        "bonferroni": "bonferroni", 
        "bonf": "bonferroni", 
        "BH": "fdr_bh", 
        "BY": "fdr_by", 
        "fdr": "fdr_bh", 
        "sidak": "sidak", 
        "holm-sidak": "holm-sidak", 
        "simes-hochberg": "simes-hochberg",
        "fdr_bh": "fdr_bh", 
        "fdr_by": "fdr_by", 
        "fdr_tsbh": "fdr_tsbh", 
        "fdr_tsbky": "fdr_tsbky", 
    }

##Prediction Settings:
DEF_MODEL = 'ridgecv'
DEF_CLASS_MODEL = 'svc'
DEF_COMB_MODELS = ['ridgecv']
DEF_FOLDS = 5
DEF_FEATURE_SELECTION_MAPPING = {
    'magic_sauce': 'Pipeline([("1_mean_value_filter", OccurrenceThreshold(threshold=int(sqrt(X.shape[0]*10000)))), ("2_univariate_select", SelectFwe(f_regression, alpha=60.0)), ("3_rpca", RandomizedPCA(n_components=max(int(X.shape[0]/max(1.5,len(self.featureGetters))), min(50, X.shape[1])), random_state=42, whiten=False, iterated_power=3))])', 
    'univariatefwe': 'SelectFwe(f_regression, alpha=60.0)',
    'pca':'RandomizedPCA(n_components=max(min(int(X.shape[1]*.10), int(X.shape[0]/max(1.5,len(self.featureGetters)))), min(50, X.shape[1])), random_state=42, whiten=False, iterated_power=3)',
    'none': None,
}

##Mediation Settings:
DEF_MEDIATION_BOOTSTRAP = 1000
DEF_OUTCOME_PATH_STARTS = []
DEF_OUTCOME_MEDIATORS = []

DEF_CORENLP_DIR = '../Tools/corenlp-python'
DEF_CORENLP_SERVER_COMMAND = './corenlp/corenlp.py'
DEF_CORENLP_PORT = 20202   #default: 20202
#CORE NLP PYTHON SERVER COMMAND (must be running): ./corenlp/corenlp.py -p 20202 -q 

TAG_RE = re.compile(r'<[^>]+>')
URL_RE = re.compile(r'(?:http[s]?\:\/\/)?(?:[\w\_\-]+\.)+(?:com|net|gov|edu|info|org|ly|be|gl|co|gs|pr|me|cc|us|uk|gd|nl|ws|am|im|fm|kr|to|jp|sg|int|mil|arpa|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|bq|br|bs|bt|bv|bw|by|bz|bzh|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cw|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zr|zw)+(?:\/[^\s ]+)?')

DEF_TOPIC_LEX_METHOD = 'csv_lik'

PERMA_SORTED = ['P+',
                'P-',
                'E+',
                'E-',
                'R+',
                'R-',
                'M+',
                'M-',
                'A+',
                'A-',
                'ope',
                'con',
                'ext',
                'agr',
                'neu'
                ]

GIGS_OF_MEMORY = 512 #used to determine when to use queries that hold data in memory
CORES = 32 #used to determine multi-processing

POSSIBLE_VALUE_FUNCS = [
    lambda d: 1,
    lambda d: float(d),
    lambda d: sqrt(float(d)),
    lambda d: log(float(d) + 1),
    lambda d: log2(float(d) + 1),
    lambda d: 2*sqrt(d+3/float(8))
]

##Meta settings
DEF_INIT_FILE = 'initFile.ini'
WARNING_STRING = "\n".join(["#"*68, "#"*68, "WARNING: %s", "#"*68, "#"*68])
##########################################################
### Static Module Methods
##
#

def alignDictsAsLists(dict1, dict2):
    """Converts two dictionaries to vectors, where the values are aligned by keys"""
    coFields = frozenset(dict1.keys()).intersection(frozenset(dict2.keys()))
    list1 = map(lambda c: dict1[c], coFields)
    #list1 = [dict1[c] for c in coFields] #equivalent to above
    list2 = map(lambda c: dict2[c], coFields)
    return (list1, list2)

def alignDictsAsXy(X, y):
    """turns a list of dicts for x and a dict for y into a matrix X and vector y"""
    keys = frozenset(y.keys())
    keys = keys.intersection(*[x.keys() for x in X])
    keys = list(keys) #to make sure it stays in order
    listy = map(lambda k: y[k], keys)
    # Order of columns of X is preserved
    listX = map(lambda k: [x[k] for x in X], keys)
    # print type(listy), type(listy[0])
    import decimal
    listy = [float(v) for v in listy] if isinstance(listy,list) and isinstance(listy[0],decimal.Decimal) else listy
    # print type(listy), type(listy[0])
    return (listX, listy)

def fiftyChecks(args):
    Xc, Xend, y, check = args
    np.random.seed()
    lr = LogisticRegression(penalty='l2', C=1000000, fit_intercept=True)
    if Xc is not None:
        r = sum([roc_auc_score(y, lr.fit(newX,y).predict_proba(newX)[:,1]) > check for newX in [np.append(Xc, permutation(Xend), 1) for i in xrange(50)]])
    else:
        # newX = permutation(Xend).reshape(len(Xend),1)
        # print type(Xend)
        # print dir(Xend)
        # print y
        # r = roc_auc_score(y, lr.fit(newX,y).predict_proba(newX)[:,1])
        r = sum([roc_auc_score(y, lr.fit(newX,y).predict_proba(newX)[:,1]) > check for newX in [permutation(Xend).reshape(len(Xend),1) for i in xrange(50)]])
    #if r: print r
    return r

def rowsToColumns(X):
    """Changes each X from being represented in columns to rows """
    return [X[0:, c] for c in xrange(len(X[0]))]

def stratifiedZScoreybyX0(X, y):
    """zscores based on the means of all unique ys"""
    #TODO: probably faster in vector operations
    #first separate all rows for each y
    X0sToYs = dict()
    for i in xrange(len(y)):
        try: 
            X0sToYs[X[i][0]].append(y[i])
        except KeyError:
            X0sToYs[X[i][0]] = [y[i]]

    #next figure out the mean for x, for each unique y
    meanYforX0s = []
    for ys in X0sToYs.itervalues():
        meanYforX0s.append(mean(ys)) #should turn into a row
        
    #figure out mean and std-dev for meanXs:
    meanOfMeans = mean(meanYforX0s) #should be a row
    stdDevOfMeans = std(meanYforX0s) #shoudl be a row

    #apply to y:
    newY = []
    for yi in y:
        newY.append((yi - meanOfMeans)/float(stdDevOfMeans))

    return (zscore(X), newY)

def switchColumnsAndRows(X):
    """Toggles X between rows of columns and columns of rows """
    if not isinstance(X, np.ndarray): X = array(X)
    return array([X[0:, c] for c in xrange(len(X[0]))])
    
def warn(string, attention=False):
    if attention: string = WARNING_STRING % string
    print >>sys.stderr, string


multSpace = re.compile(r'\s\s+')
startSpace = re.compile(r'^\s+')
endSpace = re.compile(r'\s+$')
multDots = re.compile(r'\.\.\.\.\.+') #more than four periods
newlines = re.compile(r'\s*\n\s*')
#multDots = re.compile(r'\[a-z]\[a-z]\.\.\.+') #more than four periods

def shrinkSpace(s):
    """turns multipel spaces into 1"""
    s = multSpace.sub(' ',s)
    s = multDots.sub('....',s)
    s = endSpace.sub('',s)
    s = startSpace.sub('',s)
    s = newlines.sub(' <NEWLINE> ',s)
    return s

def getGroupFreqThresh(correl_field=None):
    """set group_freq_thresh based on level of analysis"""
    group_freq_thresh = DEF_GROUP_FREQ_THRESHOLD
    if correl_field:
        if any(field in correl_field.lower() for field in ["mess", "msg"]) or correl_field.lower().startswith("id"):
            group_freq_thresh = 1
        elif any(field in correl_field.lower() for field in ["user", "usr", "auth"]):
            group_freq_thresh = 500
        elif any(field in correl_field.lower() for field in ["county", "cnty", "cty", "fips"]):
            group_freq_thresh = 40000
    return group_freq_thresh

def bonfPCorrection(tup, numFeats):
    """given tuple with second entry a p value, multiply by num of feats and return"""
    return (tup[0], tup[1]*numFeats) + tup[2:]

def pCorrection(pDict, method=DEF_P_CORR, pLevelsSimes=[0.05, 0.01, 0.001], rDict = None):
    """returns corrected p-values given a dict of [key]=> p-value"""
    method = DEF_P_MAPPING[method]
    pLevelsSimes = list(pLevelsSimes) #copy so it can be popped
    new_pDict = {}
    if method == 'simes':
        n = len(pDict)

        #pDictTuples = [[k, 1 if isnan(float(v)) else v] for k, v in pDict.iteritems()]
        pDictTuples = [[k, v] for k, v in pDict.iteritems()]
        sortDict = rDict if rDict else pDict

        sortedPTuples = sorted(pDictTuples, key=lambda tup: 0 if isnan(sortDict[tup[0]]) else fabs(sortDict[tup[0]]), reverse=True if rDict else False)
        ii = 0
        rejectRest = False
        pMax = pLevelsSimes.pop()
        for ii in xrange(len(sortedPTuples)):
            if rejectRest:
                sortedPTuples[ii][1] = 1
            else:
                newPValue = sortedPTuples[ii][1] * n / (ii + 1)
                if newPValue < pMax:
                    sortedPTuples[ii][1] = round((pMax - .00001) * pMax, 5)
                else:
                    while len(pLevelsSimes) > 0 and newPValue >= pMax:
                        pMax = pLevelsSimes.pop()
                    if len(pLevelsSimes) == 0:
                        if newPValue < pMax:
                            sortedPTuples[ii][1] = round((pMax - .00001) * pMax, 5)
                        else:
                            rejectRest = True
                            sortedPTuples[ii][1] = 1
                    else:
                        sortedPTuples[ii][1] = round((pMax - .00001) * pMax, 5)
        new_pDict = dict(sortedPTuples)
    else:
        keys = pDict.keys()
        reject, pvals_corrected, alphacSidak, alphacBonf  = mt.multipletests(pvals=pDict.values(), method=method)
        i = 0
        for key in keys:
            new_pDict[key] = pvals_corrected[i]
            i+=1
    return new_pDict

newlines = re.compile(r'\s*\n\s*')

def treatNewlines(s):
    s = newlines.sub(' <NEWLINE> ',s)
    return s

def removeNonAscii(s): 
    if s:
        new_words = []
        for w in s.split():
            if len("".join(i for i in w if (ord(i)<128 and ord(i)>20))) < len(w):
                new_words.append("<UNICODE>")
            else:
                new_words.append(w)
        return " ".join(new_words)
    return ''


def reverseDictDict(d):
    """reverses the order of keys in a dict of dicts"""
    assert isinstance(d.itervalues().next(), dict), 'reverseDictDict not given a dictionary of dictionaries'
    revDict = dict()
    for key1, subd in d.iteritems():
        for key2, value in subd.iteritems():
            if not key2 in revDict:
                revDict[key2] = dict()
            revDict[key2][key1] = value
    return revDict

def chunks(l, n):
    """ Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def permaSortedKey(s):
    if isinstance(s, (list, tuple)):
        s = str(s[0])
    if isinstance (s, (float, int)):
        return s
    if s.upper() in PERMA_SORTED:
        return PERMA_SORTED.index(s.upper())
    return s

def tupleToStr(tp):
    """Takes in a tuple and returns a string with spaces instead of commas"""
    if isinstance(tp, tuple):
        return ';'.join([tupleToStr(t) for t in tp])
    return tp

def unionDictsMaxOnCollision(d1, d2):
    """unions d1 and 2 always choosing the max between the two on a collision"""
    newD = d1.copy()
    for k2, v2 in d2.iteritems():
        if (not k2 in newD) or v2 > newD[k2]:
            newD[k2] = v2
    return newD

def rgbColorMix(fromColor, toColor, resolution, randomness = False):
    #fromColor, toColor rgb (255 max) tuple
    #resolution, how many truple to return inbetween
    #(starts at from, but comes up one short of ToColor)
    (fromColor, toColor) = (array(fromColor), array(toColor))
    fromTo = toColor - fromColor #how to get from fromColor to toColor
    fromToInc = fromTo / float(resolution)
    gradientColors = []
    for i in xrange(resolution):
        gradientColors.append(tuple([int(x) for x in round(fromColor + (i * fromToInc))]))
    if randomness: 
        for i in xrange(len(gradientColors)): 
            color = gradientColors[i]
            newcolor = []
            for value in color:
                value += 20 - randint(0, 40)
                value = max(0, min(255, value))
                newcolor.append(value)
            gradientColors[i] = tuple(newcolor)

    #print gradientColors[0:4], gradientColors[-4:] #debug
    return gradientColors


#Local maintenance:
def getReportingInt(reporting_percent, iterable_length):
    return max(1, floor(reporting_percent * iterable_length))

def report(object_description, ii, reporting_int, iterable_length):
    if ii % reporting_int == 0:
        warn("%d out of %d %s processed; %2.2f complete"%(ii, iterable_length, object_description, float(ii)/iterable_length))
