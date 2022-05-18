from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)

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
    feature_one_list = list(df[[features[0]][0]])
    feature_one_min = df[[features[0]][0]].min()
    feature_one_sum = df[[features[0]][0]].sum()

    feature_two_list = list(df[[features[1]][0]])
    feature_two_min = df[[features[1]][0]].min()
    feature_two_sum = df[[features[1]][0]].sum()

    for i in range(len(feature_one_list)):
        feature_one_list[i] = (feature_one_list[i] - feature_one_min) / feature_one_sum
        feature_two_list[i] = (feature_two_list[i] - feature_two_min) / feature_two_sum
        
    normlized_features_df = pd.DataFrame()
    normlized_features_df[features[0]] = feature_one_list
    normlized_features_df[features[1]] = feature_two_list
    
    return add_noise(normlized_features_df).to_numpy()


def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    distance = (np.linalg.norm(x-y))**2

    return distance


def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    labels = []
    for i in data:
        labels.append(np.argmin(np.sum((i.reshape((1, 2)) - centroids) ** 2, axis=1)))
    return labels


def recompute_centroids(data, k, labels):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    centroids = []
    for c in range(len(k)):
        centroids.append(np.mean([data[x] for x in range(len(data)) if labels[x] == c], axis=0))
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
    centroids = choose_initial_centroids(data, k)
    compr = True
    while(compr):
        labels = assign_to_clusters(data, centroids)
        prev_centorids = centroids
        centroids = recompute_centroids(data, centroids, labels)
        current_centroids = centroids
        if(np.array_equal(prev_centorids, current_centroids)):
            compr = False
            break
        centroids = np.array(current_centroids)
        # check this location
        return labels, centroids


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    colors = np.random.uniform(15, 80, len(centroids))
    plt.title('{0} {1}'.format("Result for kmeans witk k =", len(centroids)))
    plt.xlabel('cnt')
    plt.ylabel('hum')
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], c=colors)
    plt.show()
    # plt.savefig(path)


# def closest_centroid(data, centroids):
#     """returns an array containing the index to the nearest centroid for each point"""
#     distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
#     return np.argmin(distances, axis=0)

# def move_centroids(data, closest, centroids):
#     """returns the new centroids assigned from the points closest to them"""
#     return np.array([data[closest==k].mean(axis=0) for k in range(centroids.shape[0])])