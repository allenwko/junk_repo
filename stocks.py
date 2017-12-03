from __future__ import division
from pandas_datareader import data as pddr
import pandas as pd
import datetime
import requests
from bs4 import BeautifulSoup
import numpy as np
from scipy.stats import linregress
#%matplotlib inline

pd.options.display.max_rows = 10

def flatten_multiind(df):
    ''' Flattens multiindex '''
    df = df.copy()
    levels = len(df.columns.levels)
    new_cols = list(map(lambda x: list(df.columns.get_level_values(x)), range(levels)))
    new_cols = zip(*new_cols)
    new_cols = list(map(lambda x:"_".join(x), new_cols))
    df.columns = new_cols
    return df

def get_sp_tickers():
    ''' Scrapes list of S&P 500 companies and ticker symbols from Wikipedia'''
    
    WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    req = requests.get(WIKI_URL)
    soup = BeautifulSoup(req.content, 'lxml')
    table_classes = {"class": ["sortable", "plainrowheaders"]}
    wikitables = soup.findAll("table", table_classes)

    rows = wikitables[0].findAll('tr')
    headers = [i.text for i in rows[0].findAll('th')]
    table_data = map(lambda x:[i.text for i in x.findAll('td')], rows[1:])
    sp = pd.DataFrame(table_data, columns = headers)
    sp['Ticker symbol'] = sp['Ticker symbol'].astype(str)
    return sp

def batch_data_pull(tickers, start_date, end_date, batch_size = 200):
    '''Takes in a list of ticker symbols, and grabs all the Yahoo stock data between start and end dates'''
    
    assert len(tickers) > batch_size, 'Not a batch pull buddy'
    batches = int(round(len(tickers) / batch_size))
    ticker_batches = np.array_split(tickers, batches)
    raw_data = []
    data_source = 'yahoo'
    
    for ticker_batch in ticker_batches:
        # User pandas_reader.data.DataReader to load the desired data. As simple as that.
        panel_data = pddr.DataReader(ticker_batch, data_source, start_date, end_date)
        raw_data.append(panel_data)
        data = pd.concat(raw_data, axis=2)
    return data

def id_drop_event(stock_data, per_drop_yst):
    ''' IDs day/stock pairs when they drop greater than the per_drop_yst (%) specified. '''
    delta = (stock_data.loc['Close'] - stock_data.loc['Close'].shift(-1)) / stock_data.loc['Close'] * 100
    stack = delta.stack()
    drop_events = stack[stack < per_drop_yst]
    drop_events = drop_events.reset_index()
    drop_events['event_label'] = drop_events.index
    return drop_events

def calc_later_gains(stock_data, drop_events, later_days):
    ''' For set of ID'd events, calculate the gains at a later period '''
    days_str = "%s days" % int(later_days)
    days_td = pd.Timedelta(days_str)
    
    # filters dates to prep for weekly gains
#     after_mask = drop_events['Date'] < (stock_data['Close'].index.max() - days_td)
#     before_mask = drop_events['Date'] > (stock_data['Close'].index.min() - days_td)
#     drop_events = drop_events[after_mask & before_mask]

    # only keep dates where we have the data to calculate gains
    date_mask = (drop_events['Date'] + days_td).isin(stock_data['Close'].index)
    date_mask = date_mask & drop_events['Date'].isin(stock_data['Close'].index)
    drop_events = drop_events.loc[date_mask, :]

    drop_events.loc[:, 'date_value'] = drop_events.apply(lambda x: stock_data['Close'].loc[x['Date'], x['level_1']], axis=1)
    drop_events.loc[:, 'week_later_values'] = drop_events.apply(lambda x: stock_data['Close'].loc[x['Date'] + days_td, x['level_1']], axis=1)
    drop_events.loc[:, 'week_later_gain'] = (drop_events['week_later_values'] - \
                                     drop_events['date_value']) / drop_events['date_value'] * 100

    drop_events.rename(columns = {0:'per_drop_from_yesterday', 'level_1':'ticker'}, inplace=True)
    return drop_events

def linreg_stats(series):
    ''' To do: change range to represent any date deltas''' 
    reg = linregress(range(series.shape[0]), series)
    return reg.slope, reg.intercept

def make_vol_feats(drop_events, ts_window, stock_data):
    '''Given stock events - days and tickers - makes features based on trading volume. Requires
    length of time series window to analyze, stock data.'''
    
    # Get the day/stock combo of data you want
    volgroups = drop_events[['Date', 'ticker', 'event_label']].copy()
    vol_ts_days = 7
    volgroup_dfs = []
    # note: this method will have nulls for weekends/closed exchange days
    for day in range(1, vol_ts_days+1):
        day_vol_df = volgroups.copy()
        day_vol_df.loc[:, 'Date'] = day_vol_df.loc[:, 'Date'] - pd.Timedelta('%s days' % int(day))
        volgroup_dfs.append(day_vol_df)
    volgroups = pd.concat(volgroup_dfs)
    
    # prep stock data for merge
    vol_data = stock_data['Volume'].stack()
    vol_data = vol_data.reset_index()
    vol_data.rename(columns = {'level_1':'ticker', 0:'vol'}, inplace=True)
    
    # merge and clean data
    volgroups = volgroups.merge(vol_data, how='inner', left_on=['Date', 'ticker'], right_on=['Date', 'ticker'])
    sizes = volgroups.groupby('event_label').size()
    good_labels = sizes[sizes > 2].index # keeping events with greater than 2 data points
    volgroups = volgroups.loc[volgroups.event_label.isin(good_labels), ['event_label', 'vol']]
    
    # making features
    grouped_vols = volgroups.groupby('event_label')
    agg_dict = [np.mean,
                np.std,
                np.max,
                np.min,
                linreg_stats
               ]
    vol_feats = grouped_vols.agg({'vol':agg_dict})
    vol_feats = flatten_multiind(vol_feats)
    vol_feats['vol_slope'] = vol_feats['vol_linreg_stats'].apply(lambda x:x[0])
    vol_feats['vol_intercept'] = vol_feats['vol_linreg_stats'].apply(lambda x:x[1])
    vol_feats.drop(['vol_linreg_stats'], axis=1, inplace=True)
    vol_feats.reset_index(inplace=True)
    
    norm_cols = ['vol_std', 'vol_amax', 'vol_amin', 'vol_slope', 'vol_intercept']
    for col in norm_cols:
        vol_feats['norm_' + col] = vol_feats[col] / vol_feats['vol_mean']
    return vol_feats