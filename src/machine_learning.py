from sklearn import tree
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns

# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.callbacks import EarlyStopping

# Useful articles
# https://www.kaggle.com/code/satishgunjal/multiclass-logistic-regression-using-sklearn/notebook
# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8


def transform_price_index_to_binary(df):
    return df["price_index"].apply(lambda index: 1 if index == 1 else 0)


def get_balanced_x_y(df):
    # IMBALANCED
    df['price_index'].value_counts()
    X = df.drop('price_index', 1)  # Feature Matrix
    y = df['price_index']  # Target Variable
    # example of random oversampling to balance the class distribution
    # summarize class distribution
    print(Counter(y))
    # define oversampling strategy
    oversample = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    X, y = oversample.fit_resample(X, y)
    # summarize class distribution
    print(Counter(y))
    return X, y


def ml_analyze(model, index, title, X_train, y_train, X_test, y_test, y):
    print('### ' + str(index) + '. ' + title)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Confusion Matrix', confusion_matrix(y_test, y_pred))

    print('# Metrics:')
    print('Accuracy', model.score(X_test, y_test))
    print('F-score', f1_score(y_test, y_pred))
    print('Precision', precision_score(y_test, y_pred))
    print('Recall', recall_score(y_test, y_pred))

    print('# ROC and AUC')
    y_pred_proba = model.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # create ROC curve
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.title(title)
    plt.show()

    return [fpr, tpr, auc, title]


def logistic_regression(index, X_train, y_train, X_test, y_test, y):
    lr = LogisticRegression(random_state=0)
    measures = ml_analyze(lr, index, 'Regresja logistyczna', X_train, y_train, X_test, y_test, y)

    print('# Logistic Regression Summary')
    model = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial())
    results = model.fit()
    results.summary()

    return measures


def decision_tree(index, X_train, y_train, X_test, y_test, y):
    dt = DecisionTreeClassifier(random_state=0)
    title = 'Drzewo decyzyjne'
    measures = ml_analyze(dt, index, title, X_train, y_train, X_test, y_test, y)

    # Draw a tree
    # text_representation = tree.export_text(dt)
    # print(text_representation)
    print('# Decision tree chart')
    tree.plot_tree(dt)
    plt.show()
    plt.title(title)
    plt.savefig('../results/chart_decision_tree.png')

    return measures


def decision_tree_with_parameters(index, X_train, y_train, X_test, y_test, y):
    dt_par = DecisionTreeClassifier(random_state=0, max_depth=7, min_samples_leaf=300, min_samples_split=150)
    title = 'Drzewo decyzyjne z parametrami'
    measures = ml_analyze(dt_par, index, title, X_train, y_train, X_test, y_test, y)

    # Draw a tree
    # text_representation = tree.export_text(dt_par)
    # print(text_representation)
    print('# Decision tree chart')
    tree.plot_tree(dt_par)
    plt.show()
    plt.figure(dpi=1200)
    plt.title(title)
    plt.savefig('../results/chart_decision_tree_with_parameters.png')

    return measures


def random_forest(index, X_train, y_train, X_test, y_test, y):
    rf = RandomForestClassifier(random_state=0)

    # TODO: Draw one tree from Random Forest
    # Article: https://stackoverflow.com/questions/40155128/plot-trees-for-a-random-forest-in-python-with-scikit-learn
    #
    # fn=df.feature_names
    # cn=df.target_names
    # fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
    # tree.plot_tree(rf.estimators_[0],
    #                feature_names = fn,
    #                class_names=cn,
    #                filled = True);
    # fig.savefig('rf_individualtree.png')

    return ml_analyze(rf, index, 'Las losowy', X_train, y_train, X_test, y_test, y)


def naive_bayes(index, X_train, y_train, X_test, y_test, y):
    return ml_analyze(GaussianNB(), index, 'Naiwny klasyfikator bayesowski', X_train, y_train, X_test, y_test, y)


def knn(index, X_train, y_train, X_test, y_test, y):
    return ml_analyze(KNeighborsClassifier(), index, 'K najbliższych sąsiadów', X_train, y_train, X_test, y_test, y)


# def neural_network(X_train, y_train, X_test, y_test):
    # model = Sequential()
    # model.add(Dense(16, input_shape=(X.shape[1],), activation='relu'))  # Add an input shape! (features,)
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    # model.summary()
    #
    # # compile the model
    # model.compile(optimizer='Adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    #
    # # early stopping callback
    # # This callback will stop the training when there is no improvement in
    # # the validation loss for 10 consecutive epochs.
    # es = EarlyStopping(monitor='val_accuracy',
    #                    mode='max',  # don't minimize the accuracy!
    #                    patience=10,
    #                    restore_best_weights=True)
    #
    # # now we just update our model fit call
    # history = model.fit(X,
    #                     Y,
    #                     callbacks=[es],
    #                     epochs=80,  # you can set this to a big number!
    #                     batch_size=10,
    #                     validation_split=0.2,
    #                     shuffle=True,
    #                     verbose=1)


def machine_learning(X_train, y_train, X_test, y_test, y):
    measures = [
        logistic_regression(1, X_train, y_train, X_test, y_test, y),
        decision_tree(2, X_train, y_train, X_test, y_test, y),
        decision_tree_with_parameters(3, X_train, y_train, X_test, y_test, y),
        random_forest(4, X_train, y_train, X_test, y_test, y),
        naive_bayes(5, X_train, y_train, X_test, y_test, y),
        knn(6, X_train, y_train, X_test, y_test, y)
    ]

    for measure in measures:
        plt.plot(measure[0], measure[1], label=measure[3] + " AUC=" + str(round(measure[2], 4)))
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.title('Model comparison')
    plt.show()


# Pearson Correlation
def correlation_matrix():
    df = pd.read_csv('../data/filtered_data.csv')
    df = df.iloc[: , 1:]
    df = df.iloc[:, :-1]
    plt.figure(figsize=(20,20))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.title('Macierz korelacji Pearsona')
    plt.xticks(rotation=45, ha="right")
    plt.margins(x=0)
    plt.show()

    # Correlation with output variable
    cor_target = abs(cor["price_index"])
    #Selecting highly correlated features
    relevant_features = cor_target[cor_target>0.5]
    print(relevant_features)


# correlation_matrix()

print('###### Prepare data')
print('1. Read data')
dataframe = pd.read_csv('../data/dummy.csv')

print('2. Transform dataframe price_index to binary')
dataframe["price_index"] = transform_price_index_to_binary(dataframe)

print('3. Balance dataframe')
X, y = get_balanced_x_y(dataframe)

print('4. Split data into test set and train set')
X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.3, random_state=0)

print('###### Machine learning')
machine_learning(X_tr, y_tr, X_tst, y_tst, y)
