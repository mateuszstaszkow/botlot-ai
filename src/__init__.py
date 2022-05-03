from mongo_utils import read_mongo
from conf.properties import DB_NAME
import mapper_utils as mapper
import statistics_utils as stat
import pandas as pd

COL_FLIGHTS = 'flights'

print('1. Read DB')
flights = read_mongo(DB_NAME, COL_FLIGHTS)

print('2. Map flights')
flat_flights = mapper.map_to_flat_flights(flights)

print('3. Save CSV')
df = pd.DataFrame(flat_flights)
df.to_csv('../data/data.csv')

print('4. Load cities')
cities = mapper.get_distinct_attributes(flat_flights, 'arrival_city')
print(cities)

print('5. Load weekends')
weekends = mapper.get_distinct_attributes(flat_flights, 'weekend_startDay')
print(weekends)

print('6. Draw charts for Paris')
stat.chart_by_category(df, 'weekend_startDay')
stat.chart_by_category(df, 'date')

print('7. Assign target prices')
priced_data = stat.assign_target_price(df, cities, weekends)

print('8. Label data')
labeled_data = stat.label_data(df)

print('9. Save CSV with labeled data')
labeled_data.to_csv('../data/labeled_data.csv')

print('10. Numerify data')
numerified_data = mapper.numerify_data(labeled_data)

print('11. Save CSV with numerified data')
numerified_data.to_csv('../data/numerified_data.csv')

# In order to run machine_learning, go to the machine_learing.py
