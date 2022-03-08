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
    data = list(cursor)

    def spread(item):
        def inner_spread(i):

            return {
                'date': item['_id'],
                'total_cost': i['summary'],
                'city': i['arrival']['city'],
                'from': i['weekend']['startDay'],
                'to': i['weekend']['endDay'],
                **i
            }
        return map(inner_spread, item['data'])

    result = list(map(spread, data))
    ready_data = list()
    for element in result:
        for el in element:
            ready_data.append(el)

    df = pd.DataFrame(ready_data)
    df.to_csv('../data/data.csv')
    return df
