import socket
import logging
import itertools
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import pandas as pd
import warnings
import sys
import time

from dlatk.fwConstants import DEF_ENCODING, MYSQL_ERROR_SLEEP, MYSQL_HOST, MAX_ATTEMPTS, warn

def get_db_engine(db_schema, db_host = None, charset=DEF_ENCODING, db_config = '~/.my.cnf', port=3306):
    eng = None
    attempts = 0;
    if not db_host:
        db_host = MYSQL_HOST if socket.gethostname()=='wwbp' else 'wwbp-venti'
    while (1):
        try:
            db_url = URL(drivername='mysql', host=db_host, port=port,
                database=db_schema,
                query={
                    'read_default_file' : db_config,
                    'charset': charset
                })
            eng = create_engine(name_or_url=db_url)
            break
        except Exception as e:
            attempts += 1
            warn(" *MYSQL Connect ERROR on db:%s\n%s\n (%d attempt)"% (db, e, attempts))
            time.sleep(MYSQL_ERROR_SLEEP*attempts**2)
            if (attempts > MAX_ATTEMPTS):
                sys.exit(1)
    return eng

def create_table(db_eng, table_name, columns, more_definitions=[], table_options="", if_exists="error"):
    '''
    :param db_eng: sqlalchemy.Engine
    :param table_name: string
    :param columns: OrderedDict or dict - where keys are column names and values are column type definitions
    :param more_details: list
    :param table_options: string
    :param if_exists: string ('error' or 'replace' or 'skip')
    :return:
    '''
    res = db_eng.execute("SHOW TABLES LIKE '{}'".format(table_name))
    table_exists = res.rowcount == 1
    if table_exists and if_exists.lower() == "replace":
        db_eng.execute("DROP TABLE IF EXISTS {}".format(table_name))
    if not (table_exists and (if_exists.lower() == "skip")):
        assert type(columns.values()[0]) is str, "Values for column dict must be strings."
        if type(more_definitions) is not list:
            more_definitions = [more_definitions]
        table_def_string = ', '.join(["{} {}".format(column_name, columns[column_name]) for column_name in columns] + more_definitions)
        db_eng.execute("CREATE TABLE {} ({}) {}".format(table_name, table_def_string, table_options))

def extend_table(db_eng, table_name, new_columns, if_exists="skip"):
    '''
    Add columns to a table if they don't already exist
    :param new_columns: OrderedDict or dict - where keys are column names and values are column type definitions
    :param if_exists: string('skip') #no other options implemented yet
    '''
    if if_exists != "skip":
        raise NotImplementedError
    db_schema = db_eng.url.database
    rows = db_eng.execute("SELECT * FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = '%s' AND TABLE_NAME = '%s' AND COLUMN_NAME = '%s'" % (db_schema, table_name, new_columns.keys()[0]))
    if rows.rowcount == 0:
        print "Adding new columns to table {}...".format(table_name)
        add_column_phrase = ", ".join(["ADD COLUMN {} {}".format(column_name, new_columns[column_name]) for column_name in new_columns])
        db_eng.execute("ALTER TABLE {} {}".format(table_name, add_column_phrase))
        print "Done."
    else:
        print "Some or all of the columns exist. Ignoring."
        for row in rows:
            print row

def list_iter_to_dict_iter(list_iter, dict_keys):
    '''Take list iter + keys, return dict iter'''
    for mylist in list_iter:
        if type(mylist) is not list:
            warnings.warn("list_iter_to_dict is not getting lists, it is getting {}".format(str(type(mylist))))
        yield dict(zip(dict_keys, mylist))

def resultset_to_dict_iter(result_set):
    '''
    :param result_set: SQLAlchemy results set from db_eng.execute(sql)
    :return: iterator of dicts!
    '''
    for rp in result_set:
        yield dict(rp.items())

def select_columns(dict_iter, keys_to_keep):
    '''Take a dict iter, return a dict iter with fewer columns'''
    for mydict in dict_iter:
        yield dict((k, mydict[k]) for k in keys_to_keep)

def rename_keys(dict_iter, renamings):
    for mydict in dict_iter:
        for old_key in renamings:
            new_key = renamings[old_key]
            mydict[new_key] = mydict[old_key]
            del mydict[old_key]
        yield mydict

def mysql_insert(db_eng, db_table, dict_iterator):
    '''
    :param dict_iterator: iterator where next() yields dict that corresponds to a row of data
    :param db_eng: SQLAlchemy engine
    :param db_table: expects this to already exist
    steps through iterator, and inserts each yielded item into a mysql database
    '''
    first_row = dict_iterator.next()

    list_of_columns = first_row.keys()
    columns = ', '.join(list_of_columns)
    num_columns = len(list_of_columns)
    wildcards = ','.join(['%s']*num_columns)
    insert_statement = "INSERT INTO {0} ({1}) values({2})".format(db_table, columns, wildcards)

    _mysql_insert_row(db_eng, insert_statement, first_row, list_of_columns)
    for row_dict in dict_iterator:
        _mysql_insert_row(db_eng, insert_statement, row_dict, list_of_columns)

def _mysql_insert_row(db_eng, insert_statement, values_dict, list_of_columns):
    '''Helper function for mysql_insert'''
    values = [values_dict[column] for column in list_of_columns]
    try:
        db_eng.execute(insert_statement, values)
    except Exception as e:
        logging.warning("Error inserting data: {}{}".format(type(e), e))

def mysql_update(db_eng, db_table, dict_iterator, unique_column="id", log_every=10000):
    '''
    Updates a database table with values defined in dict iterator, finds rows using unique column
    Other fields are assumed to have the same name as their dict key
    :param unique_column: the column name for unique column, probably the id field
    '''
    first_row = dict_iterator.next()
    update_columns = first_row.keys()
    del update_columns[update_columns.index(unique_column)]
    update_statements = ", ".join(["{} = %s".format(col) for col in update_columns])
    update_sql = "UPDATE {} SET {} WHERE {} = %s".format(db_table,update_statements,unique_column)
    ###
    _mysql_update_row(db_eng, update_sql, first_row, unique_column, update_columns)
    count = 0
    starttime = time.time()
    for row in dict_iterator:
        count += 1
        if count % log_every == 0:
            time_elapsed = time.time() - starttime
            print "{} rows updated in {}s...".format(count, time_elapsed)
        _mysql_update_row(db_eng, update_sql, row, unique_column, update_columns)

def _mysql_update_row(db_eng, update_sql, row, unique_column, update_columns):
    '''Helper function for mysql_update'''
    unique_value = row[unique_column]
    #TODO - give a warning when row[column] is missing
    #warnings.warn("update data missing")
    update_values = [row[column] if column in row else None for column in update_columns]
    values = update_values + [unique_value]
    try:
        db_eng.execute(update_sql, values)
    except Exception as e:
        logging.warning("Error updating data: {}{}".format(type(e), e))

def mysql_multitable(db_eng, dict_iter, table_prefix, table_column, table_column_transform, batch_size=1000, template_table=None ):
    '''
    Takes an iterator and outputs into multiple tables based on data from the iterator
    :param db_eng: SQLAlchemy db engine
    :param dict_iter: an iterator of dicts
    :param table_prefix: for the naming of the tables
    :param table_column: column to use for deciding which table to insert into
    :param table_column_transform: function to generate table name suffix using 'table_column' value
    :param batch_size: once a target table is identified, assume the next n rows go here
    '''

    batch = list(itertools.islice(dict_iter, 0, batch_size))
    batch_num = 0
    while len(batch) > 0:
        df_batch = pd.DataFrame(batch)
        df_batch["batch_num"] = batch_num
        first_line = dict(df_batch.ix[0])
        table_name = table_prefix + table_column_transform(first_line[table_column])
        try:
            if template_table:
                db_eng.execute("CREATE TABLE IF NOT EXISTS {} LIKE {}".format(table_name, template_table))
            df_batch.to_sql(table_name, db_eng, if_exists="append", index=False)
        except Exception as e:
            exc_info = sys.exc_info()
            print "Unexpected error:", exc_info[0], exc_info[1], exc_info[2]
        else:
            print "Successful insert of {} rows!".format(len(df_batch))
        batch = list(itertools.islice(dict_iter, 0, batch_size))
        batch_num += 1


def dictify(my_iter, columns):
    for item in my_iter:
        yield dict(zip(columns, item))
