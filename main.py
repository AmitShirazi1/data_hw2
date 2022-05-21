from hashlib import new
import sys
import data as dt
import clustering as cs
import matplotlib.pyplot as plt
import numpy as np

K_VALUES = [2, 3, 5]


def main(argv):
    print("Part A: ")
    features = ["cnt", "hum"]
    df = dt.load_data(argv[1])

    dt.add_new_columns(df)
    dt.data_analysis(df)

    path = "/Users/netanelshalev/Library/CloudStorage/OneDrive-Technion/first year/ds/data_hw2/"
    # path = ""
    print("Part B: ")
    normalized_noised_array = cs.transform_data(df, features)
    for k in K_VALUES:
        print("k =", k)
        labels, centroids = cs.kmeans(normalized_noised_array, k)
        print(np.array_str(centroids, precision=3, suppress_small=True))
        path = "{0}{1}".format(path, k)
        cs.visualize_results(normalized_noised_array, labels, centroids, path)

if __name__ == '__main__':
    main(sys.argv)
