

import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth',100)

import gzip
import json
import argparse
import sys

from unidecode import unidecode


from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.dialects import mysql

from glob import glob
import os



def gzJsonFileToDf(filename):

    dfdict =[]

    for line in gzip.open(filename).readlines():
        dfdict.append(json.loads(line))

    df = pd.DataFrame.from_dict(dfdict)

    return df

def unpackJsonCols(df):
    assert type(df)== pd.core.frame.DataFrame
    for col in df:
        dictseries = df[col][df[col].map(lambda x: type(x)==dict)]
        if len(dictseries)>0:
            coldf = pd.DataFrame(dictseries.tolist())
            coldf.columns = [col+'__'+i for i in coldf.columns]
            coldf = unpackJsonCols(coldf)

            df = df.join(coldf)
            df = df.drop(col,axis=1)
    return df

def stringifyDfListCols(df):
    for col in df:
        if type(df[col][0]) == list:
            df[col] = df[col].astype(str)
            continue
    return df

def tryUnidecode(string):
    try:
        return unidecode(string)
    except:
        return string

def unidecodeDf(df):
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].apply(tryUnidecode)
            #df[col] =  df[col].str.replace('[^\x00-\x7F]','')
    return df



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='jsondir', type=str, default='/export/ws16ehr/data/qntfy/new_twitter/anonymized_unanonymized_controls')
    parser.add_argument('--glob', dest='fileglob', type=str, default='*.gz')
    parser.add_argument('-d', '--database', dest='database', type=str, default='ws16pcrut')

args = parser.parse_args()


jsondir = args.jsondir

myDB = URL(drivername='mysql', database=args.database, query={ 'read_default_file' : '~/.my.cnf' })
engine = create_engine(name_or_url=myDB, encoding="utf-8")

files = glob(os.path.join(jsondir,args.fileglob))

sqltable = os.path.split(jsondir)[-1]

# i = 0
# f = files[0]

# print f

# df = gzJsonFileToDf(f)
# df = unpackJsonCols(df)
# df = stringifyDfListCols(df)
# df = unidecodeDf(df)


sqldtypes = {u'created_at':mysql.DATETIME, u'message_id':mysql.BIGINT(20),u'message':mysql.TEXT,u'from_id':mysql.VARCHAR(20)}



for i,f in enumerate(files):
    print f
    username = os.path.basename(f).split('.')[0]
    tdf = gzJsonFileToDf(f)
    tdf = unpackJsonCols(tdf)
    tdf = tdf.loc[:,[u'created_at', u'id',u'text',u'user__screen_name']]
    tdf['created_at'] = pd.to_datetime(tdf['created_at'])
    tdf['user__screen_name'] = username
    tdf.columns = [u'created_at', u'message_id',u'message',u'from_id']
    tdf = stringifyDfListCols(tdf)
    tdf = unidecodeDf(tdf)
    tdf = tdf.set_index(['message_id','from_id'])
    #df = df.append(tdf)
    
    if i ==0:
        existsBeh = 'replace'
    else:
        existsBeh = 'append'

    tdf.to_sql(sqltable, engine, if_exists=existsBeh, chunksize=1000, dtype=sqldtypes)



