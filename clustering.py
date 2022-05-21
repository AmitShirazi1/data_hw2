from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)

LONGEST_POSSIBLE_DISTANCE = 1000000


def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.01^2)
    """
    noise = np.random.normal(loc=0, scale=0.01, size=data.shape)
    return data + noise


def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


# ====================
def transform_data(df, features):
    """
    Performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe
    :return: transformed data as numpy array of shape (n, 2)
    """

    first_feature_values = list(df[[features[0]][0]])
    first_feature_min = min(first_feature_values)
    first_feature_sum = sum(first_feature_values)

    second_feature_values = list(df[[features[1]][0]])
    second_feature_min = min(second_feature_values)
    second_feature_sum = sum(second_feature_values)

    for i in range(len(first_feature_values)):
        first_feature_values[i] = (first_feature_values[i] - first_feature_min) / first_feature_sum
        second_feature_values[i] = (second_feature_values[i] - second_feature_min) / second_feature_sum

    normalized_values_array = np.array([first_feature_values, second_feature_values]).T

    return add_noise(normalized_values_array)


# need to figure out whether it's needed
def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    distance = (np.linalg.norm(x - y)) ** 2

    return distance


def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    number_of_registries = len(data)
    labels = np.zeros((number_of_registries,), dtype=int)
    for point_index in range(number_of_registries):
        min_distance = LONGEST_POSSIBLE_DISTANCE
        for centroid_index in range(len(centroids)):
            distance_to_centroid = dist(data[point_index], centroids[centroid_index])
            if min_distance > distance_to_centroid:
                min_distance = distance_to_centroid
                labels[point_index] = centroid_index

    return labels


def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    number_of_registries = len(data)
    centroids = np.zeros(shape=(k, 2), dtype=float)
    current_cluster = np.zeros(shape=(number_of_registries, 2), dtype=float)
    for cluster_number in range(k):
        points_in_cluster_counter = 0
        for index in range(number_of_registries):
            if labels[index] == cluster_number:
                current_cluster[points_in_cluster_counter][0] = data[index][0]
                current_cluster[points_in_cluster_counter][1] = data[index][1]
                points_in_cluster_counter += 1

        centroids[cluster_number] = np.mean(current_cluster, axis=0, dtype=float)
    return centroids

def ccompute_centroids(X, idx, K):
    """Computes centroids from the mean of its cluster's members.

    Computes centroids from the mean of its cluster's members if there are
    any members for the centroid, else it returns an array of nan.

    Args:
        X (numpy.array): Features' dataset
        idx (numpy.array): Column vector of assigned centroids' indices.
        K (int): Number of centroids.

    Returns:
        numpy.array: Column vector of newly computed centroids
    """

    m, n = X.shape
    elements = None
    centroids = np.zeros((K, n))
    for k in range(K):
        elements = X[(idx == k).flatten()]
        if elements.size != 0:
            centroids[k] = np.mean(elements, axis=0, dtype=float)
        else:
            centroids[k] = np.full((1, n), np.nan, dtype=float)

    return centroids 

def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    
    for i in range(k):
        indices = np.where(idx == i)
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()
    
    return centroids 


def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """
    current_centroids = choose_initial_centroids(data, k)
    labels = np.zeros((len(data),), dtype=int)

    stabilized = False
    while not stabilized:
        labels = assign_to_clusters(data, current_centroids)
        previous_centroids = current_centroids.copy()
        current_centroids = ccompute_centroids(data, labels, k)

        if np.array_equal(previous_centroids, current_centroids):
            stabilized = True

    return labels, current_centroids


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    colors = "red"
    plt.title('{0} {1}'.format("Result for kmeans with k =", len(centroids)))
    plt.xlabel('cnt')
    plt.ylabel('hum')
    plt.scatter(centroids[:, 0], centroids[:, 1], c=colors)
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()
    # plt.savefig(path)