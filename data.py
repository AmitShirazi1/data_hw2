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


def timestamp_division(timestamp):
    months = list()
    hours = list()
    days = list()
    years = list()

    for t in timestamp:
        date_object = datetime.strptime(t, "%d/%m/%Y %H:%M")
        months.append(date_object.month)
        hours.append(date_object.hour)
        days.append(date_object.day)
        years.append(date_object.year)



        """df['Hours'] = hours
    print(df)
    # return months, hours, days, years"""


def add_new_columns(df):
    """adds columns to df and returns the new df"""
    seasons_names_array = df[["season"]].apply(seasons_naming_function, axis='columns', raw=True)
    df['season_name'] = seasons_names_array.tolist()

    """hours = df[["timestamp"]].apply(lambda t: , axis='rows', raw=True)"""

    df['Dates'] = pd.to_datetime(df['timestamp']).dt.date
    df['Time'] = pd.to_datetime(df['timestamp']).dt.time
    print(df)
    """ df['Hour'], df['Day'], df['Month'], df['Year']
    = hours.tolist()
    = days.tolist()
    df['Month'] = months.tolist()
    = years.tolist() """