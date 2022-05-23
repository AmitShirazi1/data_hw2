import pandas as pd
from datetime import datetime

SPRING = 0
SUMMER = 1
FALL = 2

REGULAR_DAY = 0
WEEKEND = 1
HOLIDAY = 2
HOLIDAY_AND_WEEKEND = 3

CORRELATION_RANKING_LIST_LENGTH = 5
CORRELATION_DIGITS_AFTER_DECIMAL_POINT = 6
SEASONS_DIGITS_AFTER_DECIMAL_POINT = 2


def load_data(path):
    """ Read and return the pandas DataFrame """
    df = pd.read_csv(path)
    return df


def seasons_naming_function(season_idx):
    """ Names every index that represents a season with the appropriate season name. """
    if season_idx == SPRING:
        return 'spring'
    elif season_idx == SUMMER:
        return 'summer'
    elif season_idx == FALL:
        return 'fall'
    return 'winter'


def timestamp_hour(timestamp):
    """ Return hour from within timestamp. """
    hour_object = datetime.strptime(timestamp[0], "%d/%m/%Y %H:%M")
    return hour_object.hour


def timestamp_day(timestamp):
    """ Return day from within timestamp. """
    day_object = datetime.strptime(timestamp[0], "%d/%m/%Y %H:%M")
    return day_object.day


def timestamp_month(timestamp):
    """ Return month from within timestamp. """
    month_object = datetime.strptime(timestamp[0], "%d/%m/%Y %H:%M")
    return month_object.month


def timestamp_year(timestamp):
    """ Return year from within timestamp. """
    year_object = datetime.strptime(timestamp[0], "%d/%m/%Y %H:%M")
    return year_object.year


def holiday_weekend_param(df):
    """ Check all given days for special days: holiday, weekend, regular day. """
    if (df[['is_holiday'][0]] == 0) and (df[['is_weekend'][0]]) == 0:
        return REGULAR_DAY
    elif (df[['is_holiday'][0]] == 0) and (df[['is_weekend'][0]]) == 1:
        return WEEKEND
    elif (df[['is_holiday'][0]] == 1) and (df[['is_weekend'][0]]) == 0:
        return HOLIDAY
    else:
        return HOLIDAY_AND_WEEKEND


def calc_t_diff(df):
    """ Calculate the difference between t2 and t1. """
    return df[['t2'][0]] - df[['t1'][0]]


def add_new_columns(df):
    """ Add columns to a given data frame, according to its current columns' values. """
    seasons_names_array = df[["season"]].apply(seasons_naming_function, axis='columns', raw=True)
    df['season_name'] = seasons_names_array.tolist()

    split_timestamp_hour = df[["timestamp"]].apply(timestamp_hour, axis=1, raw=True)
    df['Hour'] = split_timestamp_hour.tolist()

    split_timestamp_day = df[["timestamp"]].apply(timestamp_day, axis=1, raw=True)
    df['Day'] = split_timestamp_day.tolist()

    split_timestamp_month = df[["timestamp"]].apply(timestamp_month, axis=1, raw=True)
    df['Month'] = split_timestamp_month.tolist()

    split_timestamp_year = df[["timestamp"]].apply(timestamp_year, axis=1, raw=True)
    df['Year'] = split_timestamp_year.tolist()

    holiday_weekend_array = df.apply(holiday_weekend_param, axis=1)
    df['is_weekend_holiday'] = holiday_weekend_array.tolist()

    t_diff_array = df.apply(calc_t_diff, axis=1)
    df['t_diff'] = t_diff_array.tolist()


def data_analysis(df):
    """ Print statistics on a given data frame: correlation between features, difference between t1 and t2. """
    print("describe output:")
    print(df.describe().to_string())
    print()
    print("corr output:")
    corr = df.corr()
    print(corr.to_string())
    print()

    # A dict contains keys - tuples of features and values - the correlation between those features
    # for example: ('cnt', 't1' : 0.388798...)
    # The corr values in the dict are absolute.
    corr_dict = {}
    features_names = corr.columns.values.tolist()
    for i in range(len(features_names)):
        for j in range(len(features_names)):
            if i < j:
                corr_dict[(features_names[i], features_names[j])] = abs(corr[features_names[i]][features_names[j]])
    # sort the dict by values
    sorted_dict = dict(sorted(corr_dict.items(), key=lambda item: item[1]))
    # print(sorted_dict)
    sorted_list = list(sorted_dict)
    list_length = len(sorted_list)
    count = 1
    print("Highest correlated are: ")
    for i in range(list_length - 1, list_length - CORRELATION_RANKING_LIST_LENGTH - 1, -1):
        print('{0}. {1} with {2}'.format(count, sorted_list[i], "%.6f" % round(sorted_dict[sorted_list[i]],
                                                                               CORRELATION_DIGITS_AFTER_DECIMAL_POINT)))
        count += 1
    print()
    print("Lowest correlated are: ")
    for j in range(CORRELATION_RANKING_LIST_LENGTH):
        print('{0}. {1} with {2}'.format(j + 1, sorted_list[j], "%.6f" % round(sorted_dict[sorted_list[j]],
                                                                               CORRELATION_DIGITS_AFTER_DECIMAL_POINT)))

    print()
    df_by_season = df.groupby(['season_name'], as_index=True).mean()

    print("fall average t_diff is", "%.2f" % round(df_by_season[["t_diff"][0]]["fall"],
                                                   SEASONS_DIGITS_AFTER_DECIMAL_POINT))
    print("spring average t_diff is", "%.2f" % round(df_by_season[["t_diff"][0]]["spring"],
                                                     SEASONS_DIGITS_AFTER_DECIMAL_POINT))
    print("summer average t_diff is", "%.2f" % round(df_by_season[["t_diff"][0]]["summer"],
                                                     SEASONS_DIGITS_AFTER_DECIMAL_POINT))
    print("winter average t_diff is", "%.2f" % round(df_by_season[["t_diff"][0]]["winter"],
                                                     SEASONS_DIGITS_AFTER_DECIMAL_POINT))
    print("All average t_diff is", "%.2f" % round(df[["t_diff"][0]].mean(),
                                                  SEASONS_DIGITS_AFTER_DECIMAL_POINT))

