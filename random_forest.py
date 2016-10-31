# Random Forest Implementation
# See http://www.analyticbridge.com/profiles/blogs/random-forest-in-python

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

cols = list(train.iloc[[0]])
classifier = [cols[1]]
cols = cols[2:]

train_arr = train.as_matrix(cols)
train_res = train.as_matrix(classifier)

forest = RandomForestClassifier(n_estimators=100)

forest = forest.fit(train_arr, train_res)

