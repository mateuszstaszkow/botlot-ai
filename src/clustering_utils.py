from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics

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


def _map_date_to_timestamp(date):
    return time.mktime(datetime.datetime.strptime(date, '%Y-%m-%d').timetuple())


def logistic_regression(training_data, testing_data):
    y_train_labels = training_data[LABEL].values
    y_test_labels = testing_data[LABEL].values


def logistic_regression(df):
    X = df.drop('price_index', 1)   #Feature Matrix
    y = df['price_index']          #Target Variable
    #Using Pearson Correlation
    plt.figure(figsize=(12,10))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2, random_state = 1)
    lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
    lm.fit(X_train, y_train)
    pred = lm.predict(X_test)
    score = lm.score(X_test, y_test)
    print('Logit score: ', score)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Confusion Matrx')

    disp = metrics.plot_confusion_matrix(lm, X_test, y_test, display_labels= ['Low', 'Medium', 'High'], ax = ax)
    disp.confusion_matrix
    fig.show()


# 75% training, 25% testing
def split_data(df):
    return train_test_split(df, random_state=2000)


def _format_airline(airline, is_arrival):
    if 'and' in airline:
        and_index = airline.index('and')
        return airline[0:and_index - 1] if is_arrival else airline[and_index + 3:]
    return airline


def numerify_data(df):
    df = df[ALLOWED_COLUMNS]
    df['date'] = df['date'].apply(lambda d: _map_date_to_timestamp(d))
    df['weekend_startDay'] = df['weekend_startDay'].apply(_map_date_to_timestamp)
    df['arrival_airline'] = df['arrival_airline'].apply(lambda a: _format_airline(a, True))
    df['depart_airline'] = df['depart_airline'].apply(lambda a: _format_airline(a, False))
    return pd.get_dummies(df)
