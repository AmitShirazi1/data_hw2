import data as dt
import clustering as cs
import numpy as np

K_VALUES = [2, 3, 5]
DATA_FILE_CSV = "./london.csv"
FEATURES = ["cnt", "hum"]


def main():
    print("Part A: ")
    df = dt.load_data(DATA_FILE_CSV)

    dt.add_new_columns(df)
    dt.data_analysis(df)

    print()
    print("Part B: ")
    normalized_noised_array = cs.transform_data(df, FEATURES)
    line_gap = 1
    k_amount = len(K_VALUES)
    for k in K_VALUES:
        print("k =", k)
        labels, centroids = cs.kmeans(normalized_noised_array, k)
        print(np.array_str(centroids, precision=3, suppress_small=True))
        path = "./figure_k_{}.png".format(k)
        cs.visualize_results(normalized_noised_array, labels, centroids, path)
        if line_gap < k_amount:
            print()
        line_gap += 1


if __name__ == '__main__':
    main()
