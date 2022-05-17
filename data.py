from pickle import TRUE
import re
from time import time
from numpy import correlate, true_divide
import pandas as pd
from datetime import datetime


def load_data(path):
    """ reads and returns the pandas DataFrame """
    df = pd.read_csv(path)
    return df


def seasons_naming_function(season_idx):
    if season_idx == 0:
        return 'spring'
    elif season_idx == 1:
        return 'summer'
    elif season_idx == 2:
        return 'fall'
    return 'winter'


def timestamp_hour(timestamp):
    hour_object = datetime.strptime(timestamp[0], "%d/%m/%Y %H:%M")
    return hour_object.hour

def timestamp_day(timestamp):
    day_object = datetime.strptime(timestamp[0], "%d/%m/%Y %H:%M")
    return day_object.day

def timestamp_month(timestamp):
    month_object = datetime.strptime(timestamp[0], "%d/%m/%Y %H:%M")
    return month_object.month

def timestamp_year(timestamp):
    year_object = datetime.strptime(timestamp[0], "%d/%m/%Y %H:%M")
    return year_object.year

def holiday_weekend_param(df):
    if(df[['is_holiday'][0]] == 0 and df[['is_weekend'][0]] == 0):
        return 0
    elif (df[['is_holiday'][0]] == 0 and df[['is_weekend'][0]] == 1):
        return 1
    elif (df[['is_holiday'][0]] == 1 and df[['is_weekend'][0]] == 0):
        return 2
    else:
        return 3

def calc_t_diff(df):
    return df[['t2'][0]] - df[['t1'][0]]


def add_new_columns(df):
    """adds columns to df and returns the new df"""
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

    """great way without apply"""
    # df['Hour'] = pd.to_datetime(df['timestamp']).dt.hour
    # df['Day'] = pd.to_datetime(df['timestamp']).dt.day
    # df['Month'] = pd.to_datetime(df['timestamp']).dt.month
    # df['Year'] = pd.to_datetime(df['timestamp']).dt.year
    
    # print(df)
    
    
def data_analysis(df):
    """prints statistics on the transformed df"""
    print("describe output:")
    print(df.describe().to_string())
    print()
    print("corr output:")
    corr = df.corr()
    print(corr.to_string())
    print()
    
    # A dict contains keys - tuples of features and values - the correlation between those features(for example: ('cnt', 't1' : 0.388798...))
    # The corr values in the dict are absolute.
    corr_dict = {}
    features_names = corr.columns.values.tolist()
    for i in range(len(features_names)):
        for j in range(len(features_names)):
            # features_names[i] != 'timestamp' and features_names[j] != 'timestamp' and features_names[i] != 'season_name' and features_names[j] != 'season_name' and
            if(i < j):
                corr_dict[(features_names[i], features_names[j])] = abs(corr[features_names[i]][features_names[j]])
    # sort the dict by values
    sorted_dict = dict(sorted(corr_dict.items(), key=lambda item: item[1]))
    # print(sorted_dict)
    sorted_list = list(sorted_dict)
    list_length = len(sorted_list)
    count = 1
    print("Highest correlated are: ")
    for x in range(list_length-1,list_length-6,-1):
        print('{0}. {1} with {2}'.format(count, sorted_list[x] ,"%.6f" % round(sorted_dict[sorted_list[x]],6))) 
        count+=1
    print()
    print("Lowest correlated are: ")
    for x in range(5):
        print('{0}. {1} with {2}'.format(x+1, sorted_list[x] ,"%.6f" % round(sorted_dict[sorted_list[x]],6))) 
    
    print()
    df_by_season = df.groupby(['season_name'],as_index = True).mean()
    
    print("fall average t_diff is", "%.2f" %round(df_by_season[["t_diff"][0]]["fall"],2))
    print("spring average t_diff is","%.2f" % round(df_by_season[["t_diff"][0]]["spring"],2))
    print("summer average t_diff is","%.2f" % round(df_by_season[["t_diff"][0]]["summer"],2))
    print("winter average t_diff is","%.2f" % round(df_by_season[["t_diff"][0]]["winter"],2))
    print("All average t_diff is","%.2f" % round(df[["t_diff"][0]].mean(),2))