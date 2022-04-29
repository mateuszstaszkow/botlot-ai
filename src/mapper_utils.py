from datetime import datetime

DATE_MAX = datetime(2022, 9, 1)
DATE_MIN = datetime(2022, 4, 1)


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

