#!/usr/bin/env bash
#
# Runs the English PCFG parser on one or more files, printing trees only
# usage: ./lexparser.csh fileToparse+
#
# This script is meant to be called by StanfordParser.pm
#  puts a new sentence on a new line, and returns the tree structure on 
#  one line.
#
# edit by H. Andrew Schwartz

scriptdir=`dirname $0`

java -mx750m -cp "$scriptdir/stanford-parser.jar:" edu.stanford.nlp.parser.lexparser.LexicalizedParser -sentences newline -outputFormat "oneline,wordsAndTags,typedDependenciesCollapsed" $scriptdir/grammar/englishPCFG.ser.gz $*

#factored (more accurate, but slower)
#java -mx2g -cp "$scriptdir/stanford-parser.jar:" edu.stanford.nlp.parser.lexparser.LexicalizedParser -sentences newline -outputFormat "oneline,wordsAndTags,typedDependenciesCollapsed" $scriptdir/grammar/englishFactored.ser.gz $*

#kbest:
#java -mx1g -cp "$scriptdir/stanford-parser.jar:" edu.stanford.nlp.parser.lexparser.LexicalizedParser -printPCFGkBest 4 -sentences newline -outputFormat "oneline,wordsAndTags,typedDependenciesCollapsed" $scriptdir/grammar/englishPCFG.ser.gz $*

#how it's called from their sh:
#java -mx150m -cp "$scriptdir/stanford-parser.jar:" edu.stanford.nlp.parser.lexparser.LexicalizedParser \
# -outputFormat "penn,typedDependencies" $scriptdir/grammar/englishPCFG.ser.gz $*



