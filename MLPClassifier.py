#import
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

#data processing
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
le = LabelEncoder().fit(train.species)
labels = le.transform(train.species)
train = train.drop(['id', 'species'], axis=1)   #only features remain
test = test.drop(['id'], axis=1)                 #features and 'species' remain

#data processing
for i in range(len(train.values)):
    for j in range(len(train.values[i])):
        if train.values[i][j] != 0:
            train.values[i][j] = 1
y=train.values
#Using for test
sss = StratifiedShuffleSplit(labels, 1, test_size=0.2, random_state=23)
for train_index, test_index in sss:
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]


#build classifier
MyMlpClassifier=MLPClassifier(solver='adam', alpha=1e-5,
                              random_state=1)
#MyMlpClassifier=MLPClassifier(solver='lbfgs', alpha=1e-5,
                              #'''hidden_layer_sizes=(5,2)''',random_state=1)
MyMlpClassifier.fit( X_train,y_train)
Test_result=MyMlpClassifier.predict(X_test)
Test_probability=MyMlpClassifier.predict_proba(X_test)
Score=MyMlpClassifier.score(X_test,y_test)
Loss=log_loss(y_test, Test_probability)  
print "Score of MLP: {:.2%}".format(Score);
print "Log loss: {:10.4f}".format(Loss)
