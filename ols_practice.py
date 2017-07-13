##  Getting Started with the data
# 1. import all the required modules
# 2. Load the dataset
# 3. Summarize the dataset and train intuition
# 4. Visualize the dataset
# 5. Evaluate and choose algorithm
# 6. Train, predict and produce results
# 7. Analyze and prune model if needed

#   1. Import modules
import numpy as np
import pandas as pd
from sklearn import model_selection, linear_model, svm, discriminant_analysis, tree,naive_bayes, neighbors
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import preprocessing
import sklearn
from urllib.request import urlopen
import matplotlib.pyplot as plt
import matplotlib


#   2. Loading the data
url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(url,names = names)

#   3. summarize the dataset

#for shape
print(df.shape)

#Peek at the data
print(df.head(10))

#Getting statistical summary - for continuous variables
print(df.describe())

#Get class distribution - for categorical variables
#   i.e. sex
print(df.groupby('class').size())

#   4. Data visualization

#using ggplot style
matplotlib.style.use('ggplot')

# #creating box plot: There are 6 plots so we make (2,3)
# plt.figure()
# df.plot(kind = 'box', subplots = True, layout = (2,3), sharex = False, sharey = False)
# plt.show()
#
# #histogram
# plt.figure()
# df.hist()
# plt.show()

#   5. Evaluate an algorithm

#   Create a validation set
array = df.values
dim = df.shape
validation_size = 0.2 #We set 20% for validation
seed = 7

X = array[:,0:(dim[1]-2)]
y = array[:,-1]
x_train, x_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = validation_size, random_state = seed)

# We will be conducting 10-fold CV to estimate accuracy
seed = 7
scoring = 'accuracy'

#Build model_selection - in this example, we will evaluate 6 different model_selection
#   LR, LDA, KNN, CART, nb, SVm

models = []
models.append(('LR',linear_model.LogisticRegression()))
models.append(('LDA',discriminant_analysis.LinearDiscriminantAnalysis()))
models.append(('KNN', neighbors.KNeighborsClassifier()))
models.append(('CART',tree.DecisionTreeClassifier()))
models.append(('NB',naive_bayes.GaussianNB()))
models.append(('SVM',svm.SVC()))

results = []
names = []
kfold = sklearn.model_selection.KFold(n_splits = 10, random_state = seed)

for name, model in models:
    cv_results = sklearn.model_selection.cross_val_score(model, x_train, y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%r: %r (%r)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


#   Compare algorithsm
fig = plt.figure()
fig.suptitle('Algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#   Make predictions
knn = neighbors.KNeighborsClassifier()
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))
