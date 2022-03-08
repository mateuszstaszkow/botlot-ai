from mongo_utils import read_mongo
from clustering_utils import cluster_data
from conf.properties import DB_NAME
from mapper_utils import map_to_dataframe

COL_FLIGHTS = 'flights'

flights = read_mongo(DB_NAME, COL_FLIGHTS)
df = map_to_dataframe(flights)
df.to_csv('../data/data.csv')

print(flights)

# TODO: AI
# cluster_data(data)

# TODO: charts
