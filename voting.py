from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
import pandas as pd

# data preparation
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

le = LabelEncoder().fit(train.species)
labels = le.transform(train.species)
classes = list(le.classes_)
test_ids = test.id

train = train.drop(['id', 'species'], axis=1)
test = test.drop(['id'], axis=1)

train.head(1)

sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

# sklearn classifier
for train_index, test_index in sss:
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

# for voting
clf1 = KNeighborsClassifier(3)
clf2 = NuSVC(probability=True)
clf3 = LinearDiscriminantAnalysis()
clf4 = RandomForestClassifier(n_estimators=100)

classifiers = [
               RandomForestClassifier(n_estimators=100),
               DecisionTreeClassifier(),
               LinearDiscriminantAnalysis(),
               #VotingClassifier(estimators=[('kn',clf1),('nus',clf2),('lda',clf3)], voting='soft',weights=[1,1,2]),
               VotingClassifier(estimators=[('kn',clf1),('lda',clf3),('rfc',clf4)], voting='soft',weights=[2,3,1])
               ]

for clsfr in classifiers:
    clsfr.fit(X_train, y_train)
    name = clsfr.__class__.__name__
    
    print name
    
    train_predictions = clsfr.predict(X_test)
    accuracy = accuracy_score(y_test, train_predictions)
    print "Accuracy: {:.2%}".format(accuracy);
    
    train_prob = clsfr.predict_proba(X_test)
    loss = log_loss(y_test, train_prob)
    print "Log loss: {:10.4f}".format(loss)
    
    print '-'*50

# ExtraTreesClassifier was the best
votings = VotingClassifier(estimators=[('kn',clf1),('lda',clf3),('rfc',clf4)], voting='soft',weights=[1,3,1])
votings.fit(X_train,y_train)
votings_predict = votings.predict_proba(test)
#trees.fit(X_train, y_train)
#trees_predict = trees.predict_proba(test)

submission = pd.DataFrame(votings_predict, columns=classes)
submission.insert(0,'id',test_ids)
submission.to_csv('submission.csv', index=False)
