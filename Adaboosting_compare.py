import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image  #in order to show the tree
import pydotplus

n_estimators = 100  #parameters for adaboosting

#data processing
train = pd.read_csv('d:\\class_for_ucla\\2016 winter\\cs145\\project\\train.csv')
test = pd.read_csv('d:\\class_for_ucla\\2016 winter\\cs145\\project\\test.csv')
le = LabelEncoder().fit(train.species)
labels = le.transform(train.species)
train = train.drop(['id', 'species'], axis=1)   #only features remain
test = test.drop(['id'], axis=1)                 #features and 'species' remain

#divide into train data and test data


#estimators for boostclassifier
dt_stump = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1)
dt_stump.fit(train.values, labels)
dt_stump_err = 1.0 - dt_stump.score(train.values, labels)
#build determine tree in visual way
dot_data = tree.export_graphviz(
    dt_stump,
    out_file = None,
    feature_names = train._info_axis._values,
    class_names = le.classes_,
    filled = True, rounded = True,
    special_characters = True
)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")
#install graphviz to see the tree model link:http://www.graphviz.org/Download..php
Image(graph.create_png())

ada_discrete = AdaBoostClassifier(
    base_estimator =dt_stump,
    learning_rate = 1,
    n_estimators = n_estimators,
    algorithm = "SAMME"
)
ada_discrete.fit(train.values, labels)

ada_real = AdaBoostClassifier(
    base_estimator = dt_stump,
    learning_rate = 1,
    n_estimators = n_estimators,
    algorithm = "SAMME.R"
)
ada_real.fit(train.values,labels)

score_discrete = ada_discrete.score(train.values,labels)
score_real = ada_real.score(train.values,labels)

print "Accuracy of discrete: {:.2%}".format(score_discrete);
print "Accuracy of discrete: {:.2%}".format(score_real);

print 'this is the end of line'
