import numpy as np
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
    """ Select relevant features from the data, return their values after scaling and normalization.

    Keyword arguments:
    df -- dataframe as was read from the original csv.
    features -- list of 2 features from the dataframe
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


def dist(x, y):
    """ Calculate Euclidean distance between 2 vectors.

    Keyword arguments:
    x -- 1st vector
    y -- 2nd vector
    """
    distance = (np.linalg.norm(x - y)) ** 2

    return distance


def assign_to_clusters(data, centroids):
    """ Assign each data point to a cluster based on current centroids, return array of indices indicating clusters.

    Keyword arguments:
    data -- numpy array of shape (n, 2)
    centroids -- current centroids, numpy array of shape (k, 2)
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
    """ Recalculate new centroids based on the current clusters' assignment.

    Keyword arguments:
    data -- numpy array of shape (n, 2)
    labels -- current assignments to clusters for each data point, numpy array of size n
    k -- number of clusters
    """
    number_of_registries = data.shape[1]
    centroids = np.zeros((k, number_of_registries))
    for cluster_number in range(k):
        indices = np.where(labels == cluster_number)
        centroids[cluster_number, :] = (np.sum(data[indices, :], axis=1) / len(indices[0])).ravel()

    return centroids


def kmeans(data, k):
    """ Execute K-means clustering algorithm, return array of indices indicating clusters and array of their centroids.

    Keyword arguments:
    data -- numpy array of shape (n, 2)
    k -- number of clusters
    """
    current_centroids = choose_initial_centroids(data, k)
    labels = np.zeros((len(data),), dtype=int)

    stabilized = False
    while not stabilized:
        labels = assign_to_clusters(data, current_centroids)
        previous_centroids = current_centroids.copy()
        current_centroids = recompute_centroids(data, labels, k)

        if np.array_equal(previous_centroids, current_centroids):
            stabilized = True

    return labels, current_centroids


def visualize_results(data, labels, centroids, path):
    """ Visualize the K-means' data points assignment to clusters in a colorful graph, output it to a file.

    Keyword arguments:
    data -- numpy array of shape (n, 2)
    labels -- final assignment to clusters, numpy array of size n
    centroids -- final centroids' coordinates, numpy array of shape (k, 2)
    path -- path to save the output to.
    """
    plt.title("Result for kmeans with k = {}".format(len(centroids)))
    plt.xlabel('cnt')
    plt.ylabel('hum')
    plt.scatter(centroids[:, 0], centroids[:, 1])
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.savefig(path, dpi=300)
    # plt.show()
