import pandas as pd

def flatten_data(y):
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


def map_to_dataframe(flights):
    dated_flights = list(map(lambda flight: map(lambda f: {'date': flight['_id'].replace('_', '-'), **f}, flight['data']), flights))
    flat_flights = list()
    for date in dated_flights:
        for flight in date:
            flat_flight = flatten_data(flight)
            flat_flights.append(flat_flight)

    return pd.DataFrame(flat_flights)
