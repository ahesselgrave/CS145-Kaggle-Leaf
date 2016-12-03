
# coding: utf-8

# In[213]:

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[214]:

# data preparation 

le = LabelEncoder().fit(train.species)
labels = le.transform(train.species)
classes = list(le.classes_)
test_ids = test.id

train = train.drop(['id', 'species'], axis=1)
test = test.drop(['id'], axis=1)   


# In[215]:

sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

for train_index, test_index in sss:
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]


# In[216]:

# Gradient Boosting Classifier
clf = GradientBoostingClassifier(n_estimators=60, learning_rate=0.1,
      max_depth=7, random_state=0)

clf.fit(X_train, y_train)

train_predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, train_predictions)

train_probability = clf.predict_proba(X_test)
logloss = log_loss(y_test, train_probability)

print('Gradient Boosting Classifier')
print('Results')
print("Accuracy: {:.4%}".format(accuracy))
print("Log Loss: {}".format(logloss))


