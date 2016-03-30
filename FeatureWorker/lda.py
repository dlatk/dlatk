##########################################################
## lda.py
## builds a bridge between gensim and our infrastucture

##########################################################

#from gensim import corpora, models, similarities

class LDA:
    """
    This class builds a bridge between the actual lda
    implementation (TBD) and our
    infrastucture
    """
    # attributessssssss
    # contains all the words, in some form tbd
    corpus = None

    # contains the set of words used by gensim
    dictionary = None
    
    def __init__(self):
        # not important I guess ?
        print "Constructor"

    def set_params(self, nb_topics, dictionary=None, alpha=None):
        print dictionary[:20]
        self.nb_topics = nb_topics
        self.dictionary = dictionary
        self.alpha = alpha
        print 'Paraaaaameters %d' % nb_topics

    def fit(self,X,y=None):
        print 'FIT! Printing the first lines of the corpus:'
        self.corpus = [[x for x in line] for line in X]
        self.model = models.LdaModel(self.corpus,id2word=self.dictionary, num_topics=self.nb_topics, alpha=self.alpha)        

def useless():
    for i in xrange(len(docs)):
        match = re.findall(r"((#|@)?(?!(\W)\3+)([a-zA-Z\_\-\'0-9\(-\@]{2,}))",docs[i])
        docs[i] = map(lambda x: x[0],match)


    # removing stopwords
    stopwords = set(['a','able','about','across','after','all','almost','also','am','among',
                     'an','and','any','are','as','at','be','because','been','but','by','can',
                     'cannot','could','dear','did','do','does','either','else','ever','every',
                     'for','from','get','got','had','has','have','he','her','hers','him','his',
                     'how','however','i','if','in','into','is','it','its','just','least','let',
                     'like','likely','may','me','might','most','must','my','neither','no','nor',
                     'not','of','off','often','on','only','or','other','our','own','rather','said',
                     'say','says','she','should','since','so','some','than','that','the','their',
                     'them','then','there','these','they','this','tis','to','too','twas','us',
                     'wants','was','we','were','what','when','where','which','while','who',
                     'whom','why','will','with','would','yet','you','your'])
    docs = [[word for word in line if word not in stopwords] for line in docs]


    # creating dictionary                                                                                                                                                                                   
    print 'creating dictionary'
    dictionary = corpora.Dictionary(docs)

    # converting each message to a bag of words (bow)                                                                                                                                                      
    print 'converting each message to a bag of words'
    corpus = [dictionary.doc2bow(line) for line in docs]

    # creating an LDA tranformation                                                                                                                                                                         
    print 'creating an LDA tranformation'
    model = models.LdaModel(corpus,id2word=dictionary, num_topics=nb_topics, alpha=alpha)
