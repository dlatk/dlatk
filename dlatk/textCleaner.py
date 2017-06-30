import re
from . import dlaConstants as dlac

from .lib.happierfuntokenizing import Tokenizer #Potts tokenizer

### general dlatk methods
def removeNonAscii(s):
    """remove non-ascii values from string s and replace with <UNICODE>"""
    if s:
        new_words = []
        for w in s.split():
            if len("".join(i for i in w if (ord(i)<128 and ord(i)>20))) < len(w):
                new_words.append("<UNICODE>")
            else:
                new_words.append(w)
        return " ".join(new_words)
    return ''

def removeNonUTF8(s):
    """remove non-utf8 values from string s and replace with <NON-UTF8>"""
    if s:
        new_words = []
        for w in s.split():
            if len(w.encode("utf-8",'ignore').decode('utf-8','ignore')) < len(w):
                new_words.append("<NON-UTF8>")
            else:
                new_words.append(w)
        return " ".join(new_words)
    return ''

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

newlines = re.compile(r'\s*\n\s*')

def treatNewlines(s):
    s = newlines.sub(' <NEWLINE> ',s)
    return s

### method specific helper methods
# for addDedupFilterTable in messageAnnotator
def regex_or(*items):
    r = '|'.join(items)
    r = '(' + r + ')'
    return r

def pos_lookahead(r):
    return '(?=' + r + ')'

def optional(r):
    return '(%s)?' % r

mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
PunctChars = r'''['“".?!,:;]'''
Entity = '&(amp|lt|gt|quot);'
EmoticonsDN= '(:\)|:\(|:-\)|>:]|:o\)|:3|:c\)|:>|=]|8\)|=\)|:}|:^\)|>:D\)|:-D|:D|8-D|8D|x-D|xD|X-D|XD|=-D|=D|=-3|=3\)|8-\)|:-\)\)|:\)\)|>-\[|:-\(|:\(|:-c|:c|:-<|:<|:-\[|:\[|:{|>.>|<.<|>.<|:-\|\||D:<|D:|D8|D;|D=|DX|v.v|D-\':|>;\]|;-\)|;\)|\*-\)|\*\)|;-\]|;\]|;D|;^\)|>:P|:-P|:P|X-P|x-p|xp|XP|:-p|:p|=p|:-b|:b|>:o|>:O|:-O|:O|:0|o_O|o_0|o.O|8-0|>:\\|>:/|:-/|:-.|:/|:\\|=/|=\\|:S|:\||:-\||>:X|:-X|:X|:-#|:#|:$|O:-\)|0:-3|0:3|O:-\)|O:\)|0;^\)|>:\)|>;\)|>:-\)|:\'-\(|:\'\(|:\'-\)|:\'\)|;\)\)|;;\)|<3|8-}|>:D<|=\)\)|=\(\(|x\(|X\(|:-\*|:\*|:\">|~X\(|:-?)'
UrlStart1 = regex_or('https?://', r'www\.')
CommonTLDs = regex_or('com','co\\.uk','org','net','info','ca', 'co')
UrlStart2 = r'[a-z0-9\.-]+?' + r'\.' + CommonTLDs + pos_lookahead(r'[/ \W\b]')
UrlBody = r'[^ \t\r\n<>]*?'  # * not + for case of:  "go to bla.com." -- don't want period
UrlExtraCrapBeforeEnd = '%s+?' % regex_or(PunctChars, Entity)
UrlEnd = regex_or( r'\.\.+', r'[<>]', r'\s', '$')
Url = regex_or((r'[a-z0-9\.-]+?' + r'\.' + CommonTLDs + UrlEnd),(r'\b' +
    regex_or(UrlStart1, UrlStart2) +
    UrlBody +
    pos_lookahead(optional(UrlExtraCrapBeforeEnd) + UrlEnd)))

NumNum = r'\d+\.\d+'
NumberWithCommas = r'(\d+,)+?\d{3}' + pos_lookahead(regex_or('[^,]','$'))
Punct = '%s+' % PunctChars
Separators = regex_or('--+', '―')
Timelike = r'\d+:\d+h{0,1}' # removes the h trailing the hour like in 18:00h
Number = r'^\d+'
OneCharTokens = r'^.{1}$' # remove the one character tokens (maybe too agressive)
ParNumber = r'[()][+-]*\d+[()]*' # remove stuff like (+1 (-2 that appear as tokens

ExcludeThese = [
    EmoticonsDN,
    Url,
    NumNum,
    NumberWithCommas,
    Punct,
    Separators,
    Timelike,
    Number,
    OneCharTokens,
    ParNumber
]
Exclude_RE = mycompile(regex_or(*ExcludeThese))

def rttext(message):
    """
    """
    regnrt = re.compile(r"\(*RT[\s!.-:]*@\w+([\)\s:]|$)")
    regrt = re.compile(r"^RT[\s!.-:]+")
    reguser = re.compile(r"@\w+")
    regbr = re.compile(r"\[.*\]")
    regv1 = re.compile(r"\(via @\w+\)")
    regv2 = re.compile(r" via @\w+")

    rt = ''
    com = ''
    c = regnrt.search(message)
    if c:
        rt = message[c.span()[1]:].strip().strip(':').strip()
        com = message[:c.span()[0]].strip().strip(':').strip()
        if c.span()[1] == len(message):
            aux = com
            com = rt
            rt = aux
    else:
        d = regrt.search(message)
        e = reguser.search(message)
        if d and e:
            com = message[d.span()[1]:e.span()[0]]
            rt = message[e.span()[1]:]
    a = regv1.search(message)
    if not a:
        a = regv2.search(message)
    if a:
        if a.span()[0] == 0:
            b = regbr.search(message)
            rt = re.sub('^:','',message[a.span()[1]:b.span()[0]].strip()).strip()
            com = b.group()[1:len(b.group())-1]
        else:
            rt = re.sub('[|,.//]$','',message[:a.span()[0]].strip()).strip()
            com = re.sub('^:','',message[a.span()[1]:].strip()).strip()
    return rt, com

def replaceURL(message):
    message = re.sub(Url, "<URL>", message)
    message = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)', "<USER>", message)
    return message

def replaceUser(message):
    message = re.sub(r"@\w+", "<USER>", message)
    return message




def _remove_handles(text):
    """
    Remove Twitter username handles from text.
    """
    pattern = re.compile(r"(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){20}(?!@))|(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)")

    # Substitute hadnles with ' ' to ensure that text on either side of removed handles are tokenized correctly
    return pattern.sub(' ', text)

def _reduce_lengthening(text):
    """
    Replace repeated character sequences of length 3 or greater with sequences
    of length 3.
    """
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1\1", text)

def _remove_urls(text):
    pattern = re.compile(url)
    # Substitute urls with ' ' to ensure that text on either side of removed handles are tokenized correctly
    return pattern.sub(' ', text)

def sentenceNormalization(message, normalizeDict, use_unicode=dlac.DEF_UNICODE_SWITCH):


    # borrowed from CMU's Twokenize.
    # http://github.com/brendano/ark-tweet-nlp and http://www.ark.cs.cmu.edu/TweetNLP
    # Ported to Python by Myle Ott <myleott@gmail.com>.

    def regex_or(*items):
        return '(?:' + '|'.join(items) + ')'

    #  Emoticons
    # myleott: in Python the (?iu) flags affect the whole expression
    #normalEyes = "(?iu)[:=]" # 8 and x are eyes but cause problems
    normalEyes = "[:=]" # 8 and x are eyes but cause problems
    wink = "[;]"
    noseArea = "(?:|-|[^a-zA-Z0-9 ])" # doesn't get :'-(
    happyMouths = r"[D\)\]\}]+"
    sadMouths = r"[\(\[\{]+"
    tongue = "[pPd3]+"
    otherMouths = r"(?:[oO]+|[/\\]+|[vV]+|[Ss]+|[|]+)" # remove forward slash if http://'s aren't cleaned

    # mouth repetition examples:
    # @aliciakeys Put it in a love song :-))
    # @hellocalyclops =))=))=)) Oh well

    # myleott: try to be as case insensitive as possible, but still not perfect, e.g., o.O fails
    #bfLeft = u"(♥|0|o|°|v|\\$|t|x|;|\u0ca0|@|ʘ|•|・|◕|\\^|¬|\\*)".encode('utf-8')
    bfLeft = "(♥|0|[oO]|°|[vV]|\\$|[tT]|[xX]|;|\\u0ca0|@|ʘ|•|・|◕|\\^|¬|\\*)".encode('utf-8')
    bfCenter = r"(?:[\.]|[_-]+)"
    bfRight = r"\2"
    s3 = r"(?:--['\"])"
    s4 = r"(?:<|&lt;|>|&gt;)[\._-]+(?:<|&lt;|>|&gt;)"
    s5 = "(?:[.][_]+[.])"
    # myleott: in Python the (?i) flag affects the whole expression
    #basicface = "(?:(?i)" +bfLeft+bfCenter+bfRight+ ")|" +s3+ "|" +s4+ "|" + s5
    basicface = "(?:" +bfLeft+bfCenter+bfRight+ ")|" +s3+ "|" +s4+ "|" + s5

    eeLeft = r"[＼\\ƪԄ\(（<>;ヽ\-=~\*]+"
    eeRight= "[\\-=\\);'\\u0022<>ʃ）/／ノﾉ丿╯σっµ~\\*]+".encode('utf-8')
    eeSymbol = r"[^A-Za-z0-9\s\(\)\*:=-]"
    eastEmote = eeLeft + "(?:"+basicface+"|" +eeSymbol+")+" + eeRight

    oOEmote = r"(?:[oO]" + bfCenter + r"[oO])"

    emoticon = regex_or(
            # Standard version  :) :( :] :D :P
            "(?:>|&gt;)?" + regex_or(normalEyes, wink) + regex_or(noseArea,"[Oo]") + regex_or(tongue+r"(?=\W|$|RT|rt|Rt)", otherMouths+r"(?=\W|$|RT|rt|Rt)", sadMouths, happyMouths),

            # reversed version (: D:  use positive lookbehind to remove "(word):"
            # because eyes on the right side is more ambiguous with the standard usage of : ;
            regex_or("(?<=(?: ))", "(?<=(?:^))") + regex_or(sadMouths,happyMouths,otherMouths) + noseArea + regex_or(normalEyes, wink) + "(?:<|&lt;)?",

            #inspired by http://en.wikipedia.org/wiki/User:Scapler/emoticons#East_Asian_style
            eastEmote.replace("2", "1", 1), basicface,
            # iOS 'emoji' characters (some smileys, some symbols) [\ue001-\uebbb]
            # TODO should try a big precompiled lexicon from Wikipedia, Dan Ramage told me (BTO) he does this

            # myleott: o.O and O.o are two of the biggest sources of differences
            #          between this and the Java version. One little hack won't hurt...
            oOEmote
    )

    #Contractions = re.compile(u"(?i)(\w+)(n['’′]t|['’′]ve|['’′]ll|['’′]d|['’′]re|['’′]s|['’′]m)$", re.UNICODE)
    #Whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)

    punctChars = r"['\"“”‘’.?!…,:;]"
    entity     = r"&(?:amp|lt|gt|quot);"
    #  URLs

    urlStart1  = r"(?:https?://|\bwww\.)"
    commonTLDs = r"(?:com|org|edu|gov|net|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|pro|tel|travel|xxx)"
    ccTLDs   = r"(?:ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|" + \
    r"bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|" + \
    r"er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|" + \
    r"hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|" + \
    r"lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|" + \
    r"nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|" + \
    r"sl|sm|sn|so|sr|ss|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|" + \
    r"va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)"   #TODO: remove obscure country domains?
    urlStart2  = r"\b(?:[A-Za-z\d-])+(?:\.[A-Za-z0-9]+){0,3}\." + regex_or(commonTLDs, ccTLDs) + r"(?:\."+ccTLDs+r")?(?=\W|$)"
    urlBody    = r"(?:[^\.\s<>][^\s<>]*?)?"
    urlExtraCrapBeforeEnd = regex_or(punctChars, entity) + "+?"
    urlEnd     = r"(?:\.\.+|[<>]|\s|$)"
    url        = regex_or(urlStart1, urlStart2) + urlBody + "(?=(?:"+urlExtraCrapBeforeEnd+")?"+urlEnd+")"

    decorations = "(?:[♫♪]+|[★☆]+|[♥❤♡]+|[\\u2639-\\u263b]+|[\\ue001-\\uebbb]+)".encode('utf-8')
    Hearts = "(?:<+/?3+)+" #the other hearts are in decorations
    Arrows = regex_or(r"(?:<*[-―—=]*>+|<+[-―—=]*>*)", "[\\u2190-\\u21ff]+".encode('utf-8'))

    #thingsThatSplitWords = r"[^\s\.,?\"]"
    #embeddedApostrophe = thingsThatSplitWords+r"+['’′]" + thingsThatSplitWords + "*"


    ###################################################
    # remove_handles, reduce_lengthening
    ###################################################
    # from NLTK: Twitter Tokenizer
    # http://www.nltk.org/_modules/nltk/tokenize/casual.html#TweetTokenizer
    # Author: Christopher Potts <cgpotts@stanford.edu>
    #         Ewan Klein <ewan@inf.ed.ac.uk> (modifications)
    #         Pierpaolo Pantone <> (modifications)


    ### normalize text
    # normalize EOL
    message = message.replace("\n","")
    message = message.replace("\r","")

    message = re.sub(Hearts, "", message)        # remove Hearts entity, regex from CMU's Twokenize
    message = re.sub(Arrows, "", message)        # remove Arrows entity, regex from CMU's Twokenize
    message = re.sub(str(decorations), "", message)       # remove decorations entity, regex from CMU's Twokenize


    message = _remove_urls(message)       # remove URLs
    message = re.sub(r"""[\w.+-]+@[\w-]+\.(?:[\w-]\.?)+[\w-]""", "", message)        # remove email addresses, regex from NLTK Twitter Tokenizer
    message = re.sub(r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""", "", message)  # remove Twitter hashtags, regex from NLTK Twitter Tokenizer
    message = re.sub(entity, "", message)        # remove HTML entity, entity regex from CMU's Twokenize
    message = re.sub(r"""<[^>\s]+>""", "", message)      # remove HTML tags, regex from NLTK Twitter Tokenizer
    message = re.sub(r"""[\-]+>|<[\-]+""", "", message)      # remove ASCII Arrows, regex from NLTK Twitter Tokenizer
    message = re.sub(emoticon, "", message) # remove emoticon, emoticon regex from CMU's Twokenize
    message = _remove_handles(message) # regex from NLTK Twitter Tokenizer
    message = _reduce_lengthening(message) # regex from NLTK Twitter Tokenizer

    message = re.sub(r"""[`'’′]""", "'", message)  # normalize Apostrophe
    message = re.sub(r"""[“”]""", '"', message)  # normalize quotes

    # normalize EOS punctuation
    message = re.sub('\?\?+', '?', message)
    message = re.sub('\.\.+', '.', message)
    message = re.sub('\!\!+', '!', message)
    message = message.replace("?!", "?")
    message = message.replace("!?", "?")
    message = message.replace(".!", "!")
    message = message.replace("!.", ".")
    message = message.replace("’", "'")
    message = message.replace("?.", "?")
    message = message.replace(".?", "?")
    message = re.sub(r'[^\x00-\x7F]+',' ', message)

    try:
        import twokenize
        tokens = twokenize.tokenizeRawTweetText(message)
    except ImportError:
        print("warning: unable to twokenize, using happierfuntokenizing instead")
        tokenizer = Tokenizer(use_unicode)
        tokens = tokenizer.tokenize(message)

    message = ' '.join(tokens)

    for t in tokens:
        if t in normalizeDict:
            nrom_word = normalizeDict[t]
            message = message.replace(t, nrom_word)

    message = message.replace("'s", " 's")
    message = message.replace("'ve", " 've")
    message = message.replace("'ll", " 'll")
    message = message.replace("'re", " 're")
    message = message.replace("'d", " 'd")
    message = message.replace("'m", " 'm")
    message = message.replace("'M", " 'M")
    message = message.replace("'S", " 'S")
    message = message.replace("'VE", " 'VE")
    message = message.replace("'LL", " 'LL")
    message = message.replace("'RE", " 'RE")
    message = message.replace("'D", " 'D")
    message = message.replace("'M", " 'M")


    return message.strip()
