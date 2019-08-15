import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import pickle

data = pd.read_csv('car.data')

# Converting strings in data to integer representation using sklearn preprocessing
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
safety = le.fit_transform(list(data['safety']))
lug_boot = le.fit_transform(list(data['lug_boot']))
classification = le.fit_transform(list(data['class']))

predict = 'class'

x = list(zip(buying, maint, door, persons, safety, lug_boot))
y = list(classification)

x_train, x_test, y_train,  y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

"""The following section is used to train our model and select the best model"""
# best = 0
# for num_neighb in range(5, 11, 2):
#     # Splits training information from info we will use for testing
#     # Saves 10% (test_size=0.1) of the examples as x_test, y_test, so we can see how well our model predicts with new data
#     x_train, x_test, y_train,  y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#
#     model = KNeighborsClassifier(n_neighbors=num_neighb)
#     model.fit(x_train, y_train)
#
#     if model.score(x_test, y_test) > best:
#         with open('car_model.pickle', 'wb') as f:
#             pickle.dump(model, f)

# Loading the model from the pickle file
pickle_in = open('car_model.pickle', 'rb')
model = pickle.load(pickle_in)

"""The following code analyzes and prints predicted vs actual values to the console"""
# car_offer_type = ['Unacceptable', 'Acceptable', 'Good', 'Very Good']
# predicted_values = model.predict(x_test)
# total = len(predicted_values)
# correct = 0
# for i in range(total):
#     if predicted_values[i] == y_test[i]:
#         correct += 1
#         print('True', end=' ')
#     else:
#         print('False', end = ' ')
#     print('Predicted: ', car_offer_type[predicted_values[i]], 'Data: ', x_test[i], 'Actual: ', car_offer_type[y_test[i]])
# print('My score: ', correct / total)



