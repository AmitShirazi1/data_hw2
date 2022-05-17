import sys
import data as dt


def main(argv):
    df = dt.load_data(argv[1])
    dt.add_new_columns(df)
    dt.data_analysis(df)

    # print(df[['is_holiday'][0]])

if __name__ == '__main__':
    main(sys.argv)

