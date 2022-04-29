from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

LABEL = 'target_cost_medium'


def logistic_regression(training_data, testing_data):
    y_train_labels = training_data[LABEL].values
    y_test_labels = testing_data[LABEL].values


# 75% training, 25% testing
def split_data(df):
    return train_test_split(df, random_state=2000)
