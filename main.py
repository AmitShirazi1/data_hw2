from hashlib import new
import sys
import data as dt
import clustering as cs
import matplotlib.pyplot as plt
import numpy as np


def main(argv):
    # print("Part A: ")
    features = ["cnt", "hum"]
    df = dt.load_data(argv[1])
    
    # dt.add_new_columns(df)
    # dt.data_analysis(df)
    
    # labels = ""
    # path = "/Users/netanelshalev/Library/CloudStorage/OneDrive-Technion/first year/ds/data_hw2"
    path = ""
    print("Part B: ")
    new = cs.transform_data(df,features)
    # print(new)
    labels, centroids= cs.kmeans(new,5)
    # print(kmean1)
    print(np.array_str(centroids, precision=3, suppress_small=True))
    print(cs.visualize_results(new, labels, centroids, path))

    
    
if __name__ == '__main__':
    main(sys.argv)
