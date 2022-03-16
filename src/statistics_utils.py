import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

# Handle date time conversions between pandas and matplotlib
register_matplotlib_converters()

# Use white grid plot background from seaborn
sns.set(font_scale=1.5, style="whitegrid")


def chart_by_start_date_naples(data):
    # Subdataframe - Naples
    naples = data[data['arrival_city'] == 'Naples']

    # Create figure and plot space
    fig, ax = plt.subplots(figsize=(10, 10))

    # Add x-axis and y-axis
    # ax.scatter(naples['date'], naples['cost'], color='purple')

    # Set title and labels for axes
    ax.set(xlabel="Date", ylabel="Cost", title="Costs of flights to Naples")

    # Boxplot - Naples

    naples.boxplot(by='weekend_startDay', column=['summary'], grid=False, showmeans=True, ax=ax)
    # bp = plt.boxplot(data)

    plt.xticks(rotation=45, ha='right')

    plt.show()

    first_april = naples[naples['weekend_startDay'] == '2022-04-01']
    upper_quartile = np.percentile(first_april['summary'], 75)
    lower_quartile = np.percentile(first_april['summary'], 25)

    print('upper_quartile', upper_quartile)
    print('lower_quartile', lower_quartile)

def chart_by_search_date_naples(data):
    # Subdataframe - Naples
    naples = data[data['arrival_city'] == 'Naples']

    # Create figure and plot space
    fig, ax = plt.subplots(figsize=(10, 10))

    # TODO: fix
    ax.set(xlabel="Date", ylabel="Cost", title="Costs of flights to Naples")

    # Boxplot - Naples

    naples.boxplot(by='date', column=['summary'], grid=False, showmeans=True, ax=ax)

    plt.xticks(rotation=45, ha='right')

    plt.show()

    first_april = naples[naples['weekend_startDay'] == '2022-04-01']
    upper_quartile = np.percentile(first_april['summary'], 75)
    lower_quartile = np.percentile(first_april['summary'], 25)

    print('upper_quartile', upper_quartile)
    print('lower_quartile', lower_quartile)

