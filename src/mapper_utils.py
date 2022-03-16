from datetime import datetime

DATE_MAX = datetime(2022, 9, 1)


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


def map_to_flat_flights(flights):
    dated_flights = list(
        map(lambda flight:
            map(lambda f: {'date': flight['_id'].replace('_', '-'), **f}, flight['data']), flights)
    )
    flat_flights = list()
    for date in dated_flights:
        for dated_flight in date:
            flat_flight = _flatten_data(dated_flight)
            flight_date = datetime.strptime(flat_flight['weekend_startDay'], '%Y-%m-%d')
            if flight_date < DATE_MAX:
                flat_flights.append(flat_flight)
    return flat_flights


def get_cities(flights):
    cities = map(lambda flight: flight['arrival_city'], flights)
    return list(set(cities))
