import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

# Handle date time conversions between pandas and matplotlib
register_matplotlib_converters()

# Use white grid plot background from seaborn
sns.set(font_scale=1.5, style="whitegrid")

CHART_CITY = 'Paris'
EXAMPLE_DATE = '2022-06-03'
TARGET_COST_MEDIUM = 'target_cost_medium'
TARGET_COST_HIGH = 'target_cost_high'
PRICE_INDEX = 'price_index'


def _get_target_costs(city_df, city, weekend):
    weekend_df = city_df[city_df['weekend_startDay'] == weekend]
    if weekend_df.size == 0:
        return 0
    lower_quartile = np.percentile(weekend_df['summary'], 25)
    upper_quartile = np.percentile(weekend_df['summary'], 75)

    if (city == CHART_CITY) and (weekend == EXAMPLE_DATE):
        print('Example upper quartile in ' + EXAMPLE_DATE, upper_quartile)
        print('Example lower quartile in ' + EXAMPLE_DATE, lower_quartile)

    return lower_quartile, upper_quartile


def _get_price_index(row):
    summary = row['summary']
    cost_medium = row[TARGET_COST_MEDIUM]
    cost_high = row[TARGET_COST_HIGH]
    if summary <= cost_medium:
        return 1
    if (summary > cost_medium) and (summary <= cost_high):
        return 2
    else:
        return 3


def chart_by_category(data, category='weekend_startDay', city='Paris'):
    city_df = data[data['arrival_city'] == city]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(xlabel="Date", ylabel="Cost", title="Costs of flights to " + city)
    city_df.boxplot(by=category, column=['summary'], grid=False, showmeans=True, ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.show()
    plt.savefig('../results/' + city + '_' + category + '.png')


def assign_target_price(data, cities, weekends):
    data[TARGET_COST_MEDIUM] = 0
    data[TARGET_COST_HIGH] = 0
    for city in cities:
        city_df = data[data['arrival_city'] == city]
        for weekend in weekends:
            is_city_and_weekend = (data['arrival_city'] == city) & (data['weekend_startDay'] == weekend)
            medium_cost, high_cost = _get_target_costs(city_df, city, weekend)
            data.loc[is_city_and_weekend, TARGET_COST_MEDIUM] = medium_cost
            data.loc[is_city_and_weekend, TARGET_COST_HIGH] = high_cost

    return data


def label_data(priced_data):
    priced_data[PRICE_INDEX] = 0
    priced_data = priced_data.reset_index()
    for index, row in priced_data.iterrows():
        priced_data.at[index, PRICE_INDEX] = _get_price_index(row)
    return priced_data
