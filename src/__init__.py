from mongo_utils import read_mongo
from clustering_utils import cluster_data
from conf.properties import DB_NAME
import mapper_utils as mapper
import statistics_utils as stat
import pandas as pd

COL_FLIGHTS = 'flights'

flights = read_mongo(DB_NAME, COL_FLIGHTS)
flat_flights = mapper.map_to_flat_flights(flights)
cities = mapper.get_cities(flat_flights)
df = pd.DataFrame(flat_flights)
df.to_csv('../data/data.csv')

print(cities)
# TODO: labels with lower quartile
stat.chart_by_start_date_naples(df)
stat.chart_by_search_date_naples(df)

# TODO: prepare train_data 80% and test_data 20%
# TODO: logistic regression
# TODO: other methods: decision tree, random forest, CNN
# cluster_data(data)
