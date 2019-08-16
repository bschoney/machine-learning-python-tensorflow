import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer_data = datasets.load_breast_cancer()

# print(cancer_data.feature_names)
# print(cancer_data.target_names)

x = cancer_data.data
y = cancer_data.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

tumor_types = ['malignant' 'benign']

# Gettting support vector classifier
# Can give many parameters to specify how to build the hyperplane but leaving default for now
clf = svm.SVC(kernel='linear')

# Training data, building hyperplane
clf.fit(x_train, y_train)

# Getting our predicted values using the testing data
y_pred = clf.predict(x_test)

# Scoring the accuracy of our predictions
print(metrics.accuracy_score(y_test, y_pred))