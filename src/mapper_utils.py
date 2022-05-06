import datetime
import time
from datetime import datetime
import pandas as pd


DATE_MAX = datetime(2022, 9, 1)
DATE_MIN = datetime(2022, 4, 1)
LABEL = 'target_cost_medium'
ALLOWED_COLUMNS = [
    'date', # to timestamp
    'cost',
    'arrival_airline', # categories, only first value
    'arrival_startTaxiCost',
    'arrival_endTaxiCost',
    'depart_airline', # categories, only second value
    'weekend_startDay', # to timestamp
    'hotel_cost',
    'hotel_coordinates_0',
    'hotel_coordinates_1',
    'detailedFlight_start_name', # categories
    'detailedFlight_end_coordinates_0',
    'detailedFlight_end_coordinates_1',
    'price_index'
]


def _flatten_data(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


def _is_price_ok(flat_flight, attribute):
    return (attribute in flat_flight) and (flat_flight[attribute] is not None) and (flat_flight[attribute] > 0)


def _is_fully_priced(flat_flight):
    return _is_price_ok(flat_flight, 'cost') \
           and _is_price_ok(flat_flight, 'arrival_startTaxiCost') \
           and _is_price_ok(flat_flight, 'arrival_endTaxiCost') \
           and _is_price_ok(flat_flight, 'summary') \
           and _is_price_ok(flat_flight, 'hotel_cost')


def _map_date_to_timestamp(date):
    return time.mktime(datetime.strptime(date, '%Y-%m-%d').timetuple())


def _format_airline(airline, is_arrival):
    if 'and' in airline:
        and_index = airline.index('and')
        return airline[0:and_index - 1] if is_arrival else airline[and_index + 3:]
    return airline


def map_to_flat_flights(flights):
    dated_flights = list(
        map(lambda flight:
            map(lambda f: {'date': flight['_id'].replace('_', '-'), **f}, flight['data']), flights)
    )
    flat_flights = list()
    for date in dated_flights:
        for dated_flight in date:
            flat_flight = _flatten_data(dated_flight)
            date = datetime.strptime(flat_flight['weekend_startDay'], '%Y-%m-%d')
            is_date_ok = (date < DATE_MAX) and (date > DATE_MIN)
            if is_date_ok and _is_fully_priced(flat_flight):
                flat_flights.append(flat_flight)
    return flat_flights


def get_distinct_attributes(flights, attribute):
    values = map(lambda flight: flight[attribute], flights)
    distinct_values =  list(set(values))
    distinct_values.sort()
    return distinct_values


def filter_columns_and_numerify(df):
    df = df[ALLOWED_COLUMNS]
    df['date'] = df['date'].apply(lambda d: _map_date_to_timestamp(d))
    df['weekend_startDay'] = df['weekend_startDay'].apply(_map_date_to_timestamp)
    df['arrival_airline'] = df['arrival_airline'].apply(lambda a: _format_airline(a, True))
    df['depart_airline'] = df['depart_airline'].apply(lambda a: _format_airline(a, False))
    return pd.get_dummies(df)
