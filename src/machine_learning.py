import statsmodels.api as sm
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
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
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.callbacks import EarlyStopping
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler

# Useful articles
# https://www.kaggle.com/code/satishgunjal/multiclass-logistic-regression-using-sklearn/notebook
# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

df = pd.read_csv('../data/dummy.csv')
df["price_index"] = df["price_index"].apply(lambda index: 1 if index == 1 else 0)

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

# Dataset preparation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# # #Using Pearson Correlation
# plt.figure(figsize=(12,10))
# cor = df.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()

# Correlation with output variable
# cor_target = abs(cor["price_index"])
# #Selecting highly correlated features
# relevant_features = cor_target[cor_target>0.5]
# print(relevant_features)
# WNIOSEK - nic nie jest mocno skorelowane ze zmienną y

# Backward Elimination
# cols = list(X.columns)
# pmax = 1
# while (len(cols)>0):
#     p= []
#     X_1 = X[cols]
#     X_1 = sm.add_constant(X_1)
#     model = sm.OLS(y,X_1).fit()
#     p = pd.Series(model.pvalues.values[1:],index = cols)
#     pmax = max(p)
#     feature_with_p_max = p.idxmax()
#     if(pmax>0.05):
#         cols.remove(feature_with_p_max)
#     else:
#         break
# selected_features_BE = cols
# print(selected_features_BE)# from sklearn import datasets

# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

#####################
# Logistic regression
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)

# Confusion Matrix
y_pred = lr.predict(X_test)
cm_lr = confusion_matrix(y_test, y_pred)
print(cm_lr)
# [[1794 1528]
#  [1587 1730]]

# Metrics

# Accuracy
lr_accuracy = lr.score(X_test, y_test)
print(lr_accuracy)
# Accuracy = 0.5308028317517699

# F-score
f1_score_lr = f1_score(y_test, y_pred)
print(f1_score_lr)
# F-score = 0.526235741444867

# Precision
precision_lr = precision_score(y_test, y_pred)
print(precision_lr)
# Precision = 0.531000613873542

# Recall
recall_lr = recall_score(y_test, y_pred)
print(recall_lr)
# Recall = 0.5215556225504975

# Logistic Regression Summary
model = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial())
results = model.fit()
results.summary()

# ROC and AUC
# define metrics
y_pred_proba = lr.predict_proba(X_test)[::, 1]
fpr_lr, tpr_lr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc_lr = metrics.roc_auc_score(y_test, y_pred_proba)

# create ROC curve
plt.plot(fpr_lr, tpr_lr, label="AUC=" + str(auc))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

###############
# Decision tree
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)

# Confusion Matrix
y_pred = dt.predict(X_test)
cm_dt = confusion_matrix(y_test, y_pred)
print(cm_dt)
# [[3079  243]
#  [  84 3233]]

# Metrics

# Accuracy
dt_accuracy = dt.score(X_test, y_test)
print(dt_accuracy)
# Accuracy = 0.9507455942159964

# F-score
f1_score_dt = f1_score(y_test, y_pred)
print(f1_score_dt)
# F-score = 0.9518622110996614

# Precision
precision_dt = precision_score(y_test, y_pred)
print(precision_dt)
# Precision = 0.9300920598388953

# Recall
recall_dt = recall_score(y_test, y_pred)
print(recall_dt)
# Recall = 0.9746759119686463

# ROC and AUC
# define metrics
y_pred_proba = dt.predict_proba(X_test)[::, 1]
fpr_dt, tpr_dt, _ = metrics.roc_curve(y_test, y_pred_proba)
auc_dt = metrics.roc_auc_score(y_test, y_pred_proba)

# create ROC curve
plt.plot(fpr_dt, tpr_dt, label="AUC=" + str(auc))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

# Draw a tree
text_representation = tree.export_text(dt)
print(text_representation)
tree.plot_tree(dt)
plt.show()

###############################
# Decision tree with parameters
dt_par = DecisionTreeClassifier(random_state=0, max_depth=7,
                                min_samples_leaf=300, min_samples_split=150)
dt_par.fit(X_train, y_train)

# Confusion Matrix
y_pred = dt_par.predict(X_test)
cm_dt_par = confusion_matrix(y_test, y_pred)
print(cm_dt_par)
# [[2511  811]
#  [ 865 2452]]

# Metrics

# Accuracy
dt_par_accuracy = dt_par.score(X_test, y_test)
print(dt_par_accuracy)
# Accuracy = 0.7475523422202139

# F-score
f1_score_dt_par = f1_score(y_test, y_pred)
print(f1_score_dt_par)
# F-score = 0.745288753799392

# Precision
precision_dt_par = precision_score(y_test, y_pred)
print(precision_dt_par)
# Precision = 0.7514557155991419

# Recall
recall_dt_par = recall_score(y_test, y_pred)
print(recall_dt_par)
# Recall = 0.7392221887247513

# ROC and AUC
# define metrics
y_pred_proba = dt_par.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# create ROC curve
plt.plot(fpr, tpr, label="AUC=" + str(auc))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

# Draw a tree
text_representation = tree.export_text(dt_par)
print(text_representation)
tree.plot_tree(dt_par)
plt.show()
plt.figure(dpi=1200)
plt.savefig('filename.png')

###############
# Random Forest
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Confusion Matrix
y_pred = rf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred)
print(cm_rf)
# [[3079  243]
#  [  91 3226]]

# Metrics

# Accuracy
rf_accuracy = rf.score(X_test, y_test)
print(rf_accuracy)
# Accuracy = 0.9496912185570116

# F-score
f1_score_rf = f1_score(y_test, y_pred)
print(f1_score_rf)
# F-score = 0.9507810197465371

# Precision
precision_rf = precision_score(y_test, y_pred)
print(precision_rf)
# Precision = 0.9299509945229173

# Recall
recall_rf = recall_score(y_test, y_pred)
print(recall_rf)
# Recall = 0.9725655712993669

# ROC and AUC
# define metrics
y_pred_proba = rf.predict_proba(X_test)[::, 1]
fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test, y_pred_proba)
auc_rf = metrics.roc_auc_score(y_test, y_pred_proba)

# create ROC curve
plt.plot(fpr_rf, tpr_rf, label="AUC=" + str(auc))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

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

#############
# Naive Bayes
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
cm_gnb = confusion_matrix(y_test, y_pred)
print(cm_gnb)
# [[1944 1378]
#  [ 889 2428]]

# Metrics

# Accuracy
gnb_accuracy = gnb.score(X_test, y_test)
print(gnb_accuracy)
# Accuracy = 0.6585329115830697

# F-score
f1_score_gnb = f1_score(y_test, y_pred)
print(f1_score_gnb)
# F-score = 0.6817352239225045

# Precision
precision_gnb = precision_score(y_test, y_pred)
print(precision_gnb)
# Precision = 0.6379400945874935

# Recall
recall_gnb = recall_score(y_test, y_pred)
print(recall_gnb)
# Recall = 0.7319867350015073

# ROC and AUC
# define metrics
y_pred_proba = gnb.predict_proba(X_test)[::, 1]
fpr_gnb, tpr_gnb, _ = metrics.roc_curve(y_test, y_pred_proba)
auc_gnb = metrics.roc_auc_score(y_test, y_pred_proba)

# create ROC curve
plt.plot(fpr_gnb, tpr_gnb, label="AUC=" + str(auc))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

##################
# K-Nearest Neighbor
knn = KNeighborsClassifier()
y_pred = knn.fit(X_train, y_train).predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred)
print(cm_knn)
# [[2149 1173]
#  [ 653 2664]]

# Metrics

# Accuracy
knn_accuracy = knn.score(X_test, y_test)
print(knn_accuracy)
# Accuracy = 0.7249585780991114

# F-score
f1_score_knn = f1_score(y_test, y_pred)
print(f1_score_knn)
# F-score = 0.7447581772435001

# Precision
precision_knn = precision_score(y_test, y_pred)
print(precision_knn)
# Precision = 0.6942924159499609

# Recall
recall_knn = recall_score(y_test, y_pred)
print(recall_knn)
# Recall = 0.8031353632800724

# ROC and AUC
# define metrics
y_pred_proba = knn.predict_proba(X_test)[::, 1]
fpr_knn, tpr_knn, _ = metrics.roc_curve(y_test, y_pred_proba)
auc_knn = metrics.roc_auc_score(y_test, y_pred_proba)

# create ROC curve
plt.plot(fpr_knn, tpr_knn, label="AUC=" + str(auc))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

##############################
# Neural Network
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

##########################
# ROC dla wszystkich



plt.plot(fpr_knn, tpr_knn, label="KNN AUC=" + str(round(auc_knn,4)))
plt.plot(fpr_lr, tpr_lr, label="Logit, AUC=" + str(round(auc_lr,4)))
plt.plot(fpr_rf, tpr_rf, label="Las losowy, AUC=" + str(round(auc_rf,4)))
plt.plot(fpr_dt, tpr_dt, label="Drzewo decyzyjne, AUC=" + str(round(auc_dt,4)))
plt.plot(fpr_gnb, tpr_gnb, label="Bayes, AUC=" + str(round(auc_gnb,4)))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
