#!/usr/bin/env python
import sys
import gzip

#messages_file states_file >new_states_file

messageFile=open(sys.argv[1], 'rb')
stateFile=gzip.open(sys.argv[2], 'rb')

sys.stdout.write(stateFile.readline().decode())
sys.stdout.write(stateFile.readline().decode())
sys.stdout.write(stateFile.readline().decode())
error = open('error.log.txt','w+')

currentindex=-1
messageid=-1
while(True):
	line=stateFile.readline().decode()
	if len(line)==0:
		break
	tokens=line.split()
	if tokens[0]!=currentindex:
		currentindex=tokens[0]
		line_read = messageFile.readline()
		if len(line_read.split()) >  0:
			messageid=line_read.split()[0]
			if str(messageid) == '$':
				error.write("Error line: "+line_read)

	tokens[1]=messageid.decode()
	line=' '.join(tokens)
	line+='\n'
	sys.stdout.write(line)
