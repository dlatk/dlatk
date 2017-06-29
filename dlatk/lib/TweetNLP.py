#!/usr/bin/python
###########################################
## TweetNLP.py
##
## interface with CMU twitter tokenizer / pos tagger
##
## ./runTagger.sh -input example_tweets.txt -output tagged_tweets.txt
##
## hansens@seas.upenn.edu

import argparse
from subprocess import check_call, Popen, PIPE
from pprint import pprint
import os

##DEFAULTS:
#_DefaultDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # for code release
_DefaultDir = '/home/hansens'
_DefaultParams ={
    'tagger_dir' : _DefaultDir + '/Tools/TwitterTagger/ark-tweet-nlp-0.3',
    'tagger_command' : './runTagger.sh',
    'tagger_args' : ['--input-format', 'text', '--output-format', 'pretsv'],
    'tokenizer_command' : './twokenize.sh',
    'tokenizer_args' : ['--input-format', 'text', '--output-format', 'pretsv'],
    };

class TweetNLP:

    def __init__(self, **kwargs):
        self.__dict__.update(_DefaultParams)
        self.__dict__.update(kwargs)
        if not isinstance(self.tagger_args, list):
            self.tagger_args = eval(self.tagger_args)
        if not isinstance(self.tokenizer_args, list):
            self.tokenizer_args = eval(self.tokenizer_args)
        self.taggerP = None
        self.tokenizerP = None


    ###TAGGER###

    def getTaggerProcess(self):
        #start the server:
        if not self.taggerP:
            command = self.tagger_dir+'/'+self.tagger_command
            self.taggerP =  Popen([command]+self.tagger_args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        return self.taggerP

        
    def tag(self, sents):
        """returns a list of lists of tuples (word, tag) """
        """List is in the same order as the sents"""
        if sents:
            givenList = isinstance(sents, list)
            if not givenList: sents = [sents]

            taggerP = self.getTaggerProcess()
            taglists = []

            #call the tagger on each input:
            for s in sents:
                if s:
                    s = s.strip().replace("\n", '  ') + "\n"
                    taggerP.stdin.write(s.encode('utf-8'))
                    taggerP.stdin.flush()
                    tagLine = taggerP.stdout.readline().strip()
                    if tagLine:
                        info = tagLine.decode().split('\t')
                        (tokens, tags, probs) =  [l.split() for l in info[:3]]
                        taglists.append({'tokens':tokens, 'tags':tags, 'probs':probs, 'original':info[3]})
                else:
                        taglists.append({'tokens':[], 'tags':[], 'probs':[], 'original':s})

            if givenList:
                return taglists
            else:
                if taglists:
                    return taglists[0]
                return None
        return None


    ###TOKENIZER###

    def getTokenizerProcess(self):
        #start the server:
        if not self.tokenizerP:
            command = self.tagger_dir+'/'+self.tokenizer_command
            self.tokenizerP =  Popen([command]+self.tokenizer_args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        return self.tokenizerP

        
    def tokenize(self, sents):
        """returns a list of lists of tuples (word, tag) """
        """List is in the same order as the sents"""
        givenList = isinstance(sents, list)
        if not givenList: sents = [sents]
        
        tokenizerP = self.getTokenizerProcess()
        tokenizedLists = []

        #call the tokenizer on each input:
        for s in sents:
            s = s.strip().replace("\n", '  ') + "\n"
            tokenizerP.stdin.write(s.encode('utf8'))
            tokenizerP.stdin.flush()
            tagLine = tokenizerP.stdout.readline().strip()
            if tagLine:
                info = tagLine.decode().split('\t')
                tokens =  info[0].split()
                tokenizedLists.append(tokens)

        if givenList:
            return tokenizedLists
        else:
            return tokenizedLists[0]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test the TweetNLP python interface.', prefix_chars='-+', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for param, value in _DefaultParams.items():
        parser.add_argument('--'+str(param), metavar='string', dest=str(param), default=str(value),
                        help="%s default param" % param)

    parser.add_argument('-t', '--tag', metavar='string', dest='taglines', nargs='+', default=[],
                        help="lines to parse (place each in quotes)")

    parser.add_argument('-o', '--tokenize', metavar='string', dest='tokenlines', nargs='+', default=[],
                        help="lines to parse (place each in quotes)")

    args = parser.parse_args()

    if args.taglines:
        pprint(TweetNLP(**args.__dict__).tag(args.taglines))
    if args.tokenlines:
        pprint(TweetNLP(**args.__dict__).tokenize(args.tokenlines))

