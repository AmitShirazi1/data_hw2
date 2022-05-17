from pickle import TRUE
import re
from time import time
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
    return df[['t1'][0]] - df[['t2'][0]]


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
    
    print(df)
    