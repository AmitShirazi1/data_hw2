import sys
import data as dt
import clustering as cs

def main(argv):
    # print("Part A: ")
    features = ["cnt", "hum"]
    df = dt.load_data(argv[1])
    # dt.add_new_columns(df)
    # dt.data_analysis(df)
    print("Part B: ")

    cs.transform_data(df,features)
if __name__ == '__main__':
    main(sys.argv)

