from mongo_service import read_mongo
from clustering_utils import cluster_data
from conf.properties import DB_NAME

COL_FLIGHTS = 'flights'

data = read_mongo(DB_NAME, COL_FLIGHTS)
print(data)

# TODO: AI

# TODO: charts
