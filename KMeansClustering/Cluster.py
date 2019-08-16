import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# Represents number of clusters corresponding to digits 0-9
k = 10

# Loading the data from the sklearn library
digits = load_digits()
y = digits.target
# Reduces outliers and decreases computation time
data = scale(digits.data)

# Gets all the samples and all the features from each sample
samples, features = data.shape

# Evaluates the clusters and prints the results
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_, metric='euclidean')))

# Getting a KMeans object and configuring it to match num clusters we want
classifier = KMeans(n_clusters=k, init='k-means++', n_init=20)
# Calling function to fit data and print results for 25 generations
for generation in range(25):
    bench_k_means(classifier, str(generation), data)