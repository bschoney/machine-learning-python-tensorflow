import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style

# Reads in data from csv file with ; as delimiter
data = pd.read_csv("student-mat.csv", sep=';')

# Retrieves the attributes we want from the data set
data = data[['age', 'G1', 'G2', 'G3', 'studytime', 'failures', 'freetime', 'health', 'absences']]

# Attribute we will predict
predict = 'G3'

# Makes numpy array of data without column we will predict
x = np.array(data.drop([predict], 1))
# Makes numpy array of the values we want to predict
y = np.array(data[predict])

# Splits training information from info we will use for testing
# Saves 10% (test_size=0.1) of the examples as x_test, y_test, so we can see how well our model predicts with new data
x_train, x_test, y_train,  y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

"""Following commented section trains and saves best model"""
# best_model = 0
# for _ in range(50):
#
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#
#     # Getting a linear model object from sklearn
#     linear = linear_model.LinearRegression()
#
#     # Creating line of best fit using training data
#     linear.fit(x_train, y_train)
#
#     # Testing the model using our test data and printing the accuracy
#     acc = linear.score(x_test, y_test)
#
#     if acc > best_model:
#         best_model = acc
#         # Saving model into a pickle file
#         with open('student_model.pickle', 'wb') as f:
#             pickle.dump(linear, f)

# Loading the model from the pickle file
pickle_in = open('student_model.pickle', 'rb')
linear = pickle.load(pickle_in)

# Testing the model using our test data and printing the accuracy
acc = linear.score(x_test, y_test)
print(acc)

print('Coefficients: ', linear.coef_)
print('Intercept: ', linear.intercept_)

# Visualizing the model
# style.use('ggplot')
# # indep is our x-axis
# indep = 'absences'
# # makes a scatter plot (of dots on xy axis) where indep is x-axis and the thing we predict is y-axis
# pyplot.scatter(data[indep], data[predict])
# # labeling x and y axes
# pyplot.xlabel(indep)
# pyplot.ylabel('Final Grade')
# pyplot.show()