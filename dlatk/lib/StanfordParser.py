#!/usr/bin/python
###########################################
## StanfordParser.py
##
## requires the following to be in a shell script, usually named oneline.sh
## in the parser directory: 
##
## #!/usr/bin/env bash
## scriptdir=`dirname $0`
## java -mx750m -cp "$scriptdir/stanford-parser.jar:" edu.stanford.nlp.parser.lexparser.LexicalizedParser -sentences newline -outputFormat "oneline,wordsAndTags,typedDependenciesCollapsed" $scriptdir/grammar/englishPCFG.ser.gz $*


import sys
import os
import argparse
import subprocess
import random
import re
from pprint import pprint
import glob

##DEFAULTS FOR RELEASED VERSION
#_InstallPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/Tools/StanfordParser/' # folder
#_InstallDir = glob.glob(_InstallPath + "stanford-parser-full*")[0]

##DEFAULTS:
_InstallPath = '/home/hansens/Tools/StanfordParser/' # folder
_InstallDir = 'stanford-parser-2012-02-03'

_DefaultParams ={
    'save_file' : 'parsed.data',
    'save_dir' : 'backupParses',
    'parser_dir' : _InstallPath + _InstallDir,
    'parser_command' : 'oneline.sh', #note this was an edited lexparser.csh to include oneline output option to make the tree easier to handle
    'temp_file' : _InstallPath + 'temp.file', #holds sentences to be parsed
    'max_sent_words' : int(60),
    #'split' : 0,#whether to ask the parse to split sentences.(NOTE: when on, also sets a maximum sentence size)
    };

class StanfordParser:

    def __init__(self, **kwargs):
        self.__dict__.update(_DefaultParams)
        self.__dict__.update(kwargs)
        
    def parse(self, sents):
        """returns a list of dicts with parse information: const, dep, and pos """
        """List is in the same order as the sents"""

        #create temp fileName
        tempFileName = self.temp_file+'.'+str(random.randint(1, 1000000))
        
        #write the sents to a file
        f = open(tempFileName,'w')
        for s in sents:
            s = shortenToNWords(s.strip().replace("\n", ' '), int(self.max_sent_words))
            f.write(s)
            f.write("\n")
        f.close()

        #call the parser
        command = self.parser_dir+'/'+self.parser_command
        p = subprocess.Popen([command, tempFileName], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, errors = p.communicate()

        #parse the results
        #print output #debug
        parseDicts = self.getParseDicts(output)
        if not len(parseDicts) == len(sents):
            for i in range(min(len(parseDicts), len(sents))):
                pprint((sents[i], parseDicts[i]['pos']))
                print("\n")
            print("Number of parses does not match number of sents")
            print(" number of sents:  %d" % len(sents))
            print(" number of parses: %d" % len(parseDicts))
            print(sys.stderr.write("!!ERROR!! sents and parses do not match")) 
            sys.exit(0)

        os.remove(tempFileName)

        return parseDicts

    depParseRe = re.compile(r'^[a-z0-9\_]+\([a-z0-9]', re.I)
    def getParseDicts(self, output):
        parseDicts = []
        lines = output.decode().split("\n")
        i = 0
        while i < len(lines):
            parse = dict()
            #get pos
            while i < len(lines) and not lines[i]:
                i+=1
            if i >= len(lines): break
            parse['pos'] = lines[i].strip()
            i+=1

            #get const
            while i < len(lines) and not lines[i]:
                i+=1
            if i >= len(lines): break
            parse['const'] = lines[i].strip()
            i+=1

            #get dep
            depLines = []
            while i < len(lines) and not lines[i]:
                i+=1
            while i < len(lines) and lines[i]:
                depLines.append(lines[i].strip())
                i+=1
            if i < len(lines) and len(depLines) < 2 and not StanfordParser.depParseRe.match(depLines[0]):
                i-=1
                depLines = []
            parse['dep'] = depLines

            parseDicts.append(parse)

        return parseDicts


def shortenToNWords(sent, n):
    words = sent.split()
    return ' '.join(words[:int(n)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test the stanford parser python interface.', prefix_chars='-+', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for param, value in _DefaultParams.items():
        parser.add_argument('--'+str(param), metavar='string', dest=str(param), default=str(value),
                        help="%s default param" % param)

    parser.add_argument('-p', '--parse', metavar='string', dest='parselines', nargs='+', default=[],
                        help="lines to parse (place each in quotes)")

    args = parser.parse_args()

    if args.parselines:
        pprint(StanfordParser(**args.__dict__).parse(args.parselines))

