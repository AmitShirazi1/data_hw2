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

def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """
    labels = np.array()
    # Initialize centroids and error
    centroids = choose_initial_centroids(data, k)
    error = []
    compr = True
    i = 0
    prev_centorids = []
    # np.array_equal(prev_centorids, current_centroids == True
    while(compr):
        # Obtain centroids and error

        return labels, centroids
    
    
def calculate_error(a,b):
    '''
    Given two Numpy Arrays, calculates the root of the sum of squared errors.
    '''
    error = np.square(np.sum((a-b)**2))
    
    return error    
    
def assign_centroid(data, centroids):
    '''
    Receives a dataframe of data and centroids and returns a list assigning each observation a centroid.
    data: a dataframe with all data that will be used.
    centroids: a dataframe with the centroids. For assignment the index will be used.
    '''

    n_observations = data.shape[0]
    centroid_assign = []
    centroid_errors = []
    k = centroids.shape[0]


    for observation in range(n_observations):

        # Calculate the errror
        errors = np.array([])
        for centroid in range(k):
            error = calculate_error(centroids.iloc[centroid, :2], data.iloc[observation,:2])
            errors = np.append(errors, error)

        # Calculate closest centroid & error 
        closest_centroid =  np.where(errors == np.amin(errors))[0].tolist()[0]
        centroid_error = np.amin(errors)

        # Assign values to lists
        centroid_assign.append(closest_centroid)
        centroid_errors.append(centroid_error)

    return (centroid_assign,centroid_errors)

def knn(data, k):
    '''
    Given a dataset and number of clusters, it clusterizes the data. 
    data: a DataFrame with all information necessary
    k: number of clusters to create
    '''

    # Initialize centroids and error
    # centroids = initialize_centroids(data, k)
    error = []
    compr = True
    i = 0

    while(compr):
        # Obtain centroids and error
        data['centroid'], iter_error = assign_centroid(data,centroids)
        error.append(sum(iter_error))
        # Recalculate centroids
        centroids = data.groupby('centroid').agg('mean').reset_index(drop = True)

        # Check if the error has decreased
        if(len(error)<2):
            compr = True
        else:
            if(round(error[i],3) !=  round(error[i-1],3)):
                compr = True
            else:
                compr = False
        i = i + 1 

    data['centroid'], iter_error = assign_centroid(data,centroids)
    centroids = data.groupby('centroid').agg('mean').reset_index(drop = True)
    return (data['centroid'], iter_error, centroids)

def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    colors = {0:'red', 1:'blue', 2:'green'}
    plt.scatter(data.iloc[:,0], data.iloc[:,1],  marker = 'o', c = data['centroid'].apply(lambda x: colors[x]), alpha = 0.5)
    plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],  marker = 'o', s=300, 
    c = centroids.index.map(lambda x: colors[x]))
    plt.savefig(path)


def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    distance = np.linalg.norm(x-y)

    return distance


def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    
    n_observations = data.shape[0]
    centroid_errors = []
    centroid_assign = []
    # k- num of centorids ? 
    k = centroids.shape[0]

    for observation in range(n_observations):
        # Calculate the error
        errors = np.array([])
        for centroid in range(k):
            error = calculate_error(centroids[centroid, :2], data[observation,:2])
            errors = np.append(errors, error)
            
        # Calculate closest centroid & error 
        closest_centroid =  np.where(errors == np.amin(errors))[0].tolist()[0]
        centroid_error = np.amin(errors)

        # Assign values to lists
        centroid_assign.append(closest_centroid)
        centroid_errors.append(centroid_error)

    print (centroid_errors)
    return (centroid_assign,centroid_errors)


    # return labels


def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    pass
    # return centroids

