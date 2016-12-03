import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from IPython.display import Image  #in order to show the tree
import pydotplus

n_estimators = 100  #parameters for adaboosting

#data processing
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
le = LabelEncoder().fit(train.species)
labels = le.transform(train.species)
train = train.drop(['id', 'species'], axis=1)   #only features remain
test = test.drop(['id'], axis=1)                 #features and 'species' remain

#divide into train data and test data
sss = StratifiedShuffleSplit(labels, 1, test_size=0.1, random_state=23)
for train_index, test_index in sss:
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

#estimators for boostclassifier
dt_stump = DecisionTreeClassifier(max_depth=15, min_samples_leaf=1)
dt_stump.fit(X_train, y_train)
dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)
#build determine tree in visual way
dot_data = tree.export_graphviz(
    dt_stump,
    out_file = None,
    feature_names = train._info_axis._values,
    class_names = le.classes_,
    filled = True, rounded = True,
    special_characters = True
)

#graph = pydotplus.graph_from_dot_data(dot_data)
#graph.write_pdf("iris.pdf")
#install graphviz to see the tree model link:http://www.graphviz.org/Download..php
#Image(graph.create_png())

ada_real = AdaBoostClassifier(
    base_estimator = dt_stump,
    learning_rate = 1,
    n_estimators = n_estimators,
    algorithm = "SAMME.R"
)
ada_real.fit(X_train,y_train)

score_real = ada_real.score(X_test,y_test)
train_prob = ada_real.predict_proba(X_test)
loss = log_loss(y_test, train_prob)

print "Accuracy of AdaBoost: {:.2%}".format(score_real)
print "Log loss: {:10.4f}".format(loss)

