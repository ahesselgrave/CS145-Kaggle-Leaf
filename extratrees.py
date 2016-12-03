from scipy.stats import uniform
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import accuracy_score, log_loss
import pandas as pd


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

le = LabelEncoder().fit(train.species)
labels = le.transform(train.species)
classes = list(le.classes_)
test_ids = test.id


train = train.drop(['id', 'species'], axis=1)
test = test.drop(['id'], axis=1)

sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

for train_index, test_index in sss:
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

trees = ExtraTreesClassifier(
    n_estimators=100,
    max_features=None,
    min_samples_split=1
)
trees.fit(X_train, y_train)

train_predictions = trees.predict(X_test)
accuracy = accuracy_score(y_test, train_predictions)
print "Accuracy: {:.2%}".format(accuracy)

train_prob = trees.predict_proba(X_test)
loss = log_loss(y_test, train_prob)
print "Log loss: {:10.4f}".format(loss)

trees_predict = trees.predict_proba(test)

submission = pd.DataFrame(trees_predict, columns=classes)
submission.insert(0,'id',test_ids)
submission.to_csv('submission.csv', index=False)

