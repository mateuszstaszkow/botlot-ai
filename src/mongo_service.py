import pandas as pd
from pymongo import MongoClient
from conf.properties import DB_NAME, DB_USERNAME, DB_PASSWORD


def _connect_mongo(db):
    mongo_uri = 'mongodb+srv://%s:%s@%s.tsdit.mongodb.net/myFirstDatabase?retryWrites=true&w=majority' \
                % (DB_USERNAME, DB_PASSWORD, DB_NAME)
    conn = MongoClient(mongo_uri)
    return conn[db]


def read_mongo(db, collection, query={}):
    db = _connect_mongo(db)
    cursor = db[collection].find(query)
    return pd.DataFrame(list(cursor))
