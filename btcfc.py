import pandas as pd
df = pd.read_csv("BTC 18Jul2010-14May2021.csv", parse_dates=True, index_col='Date',)
df = df[['Close']]
df = df.loc[::-1] 

#import locale
#from locale import atof
#locale.setlocale(locale.LC_NUMERIC, '')
#df = df.applymap(atof)

df['Close'] = df['Close'].str.replace(",","").astype(float)

import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
START = "2021-05-16"
TODAY = date.today().strftime("%Y-%m-%d")

@st.cache
def load_data(ticker):
    yf_data = yf.download(ticker, START, TODAY)
    yf_data.reset_index(inplace=True)
    return yf_data

data_load_state = st.text('Loading data...')
yf_data = load_data('BTC-USD')
data_load_state.text('Loading data... done!')

yf_data = yf_data.set_index('Date')
yf_data = yf_data[['Close']]

df = df.append(yf_data)

df.plot()

#2012 cycle using previous record high
#df[0] - 18 Jul 2010
#df[864] - 28 Nov 2012

rec_high_value_before_2012_halving = df[0:865]['Close'].max()
rec_high_date_before_2012_halving = df[0:865]['Close'].idxmax()

rec_high_value_before_2012_halving, rec_high_date_before_2012_halving

from datetime import datetime, timedelta
halving2012 = datetime.strptime('28/11/12', '%d/%m/%y') 
halving2016 = datetime.strptime('9/7/16', '%d/%m/%y')
halving2020 = datetime.strptime('11/5/20', '%d/%m/%y')

df = df.loc[halving2012:].copy()
base2020 = df.loc[halving2020]['Close']
base2016 = df.loc[halving2016]['Close']
base2012 = df.loc[halving2012]['Close']

df['cyclebase'] = 0

df.loc[halving2012:halving2016, 'cyclebase'] = base2012
df.loc[halving2016:halving2020, 'cyclebase'] = base2016
df.loc[halving2020:, 'cyclebase'] = base2020

df['change'] = df['Close'] / df['cyclebase'] 

import numpy as np

index2012 = np.where(df.index==halving2012)[0][0]
index2016 = np.where(df.index==halving2016)[0][0]
index2020 = np.where(df.index==halving2020)[0][0]

df2012 = df[index2012:index2016].copy()
df2012 = df2012.reset_index()

df2016 = df[index2016:index2020].copy()
df2016 = df2016.reset_index()

df2020 = df[index2020:].copy()
df2020 = df2020.reset_index()

x_max = max(index2016, index2020-index2016)
# Data
import matplotlib.pyplot as plt 
df1=pd.DataFrame({'# of Days': range(x_max), '2012': df2012['change'], '2016': df2016['change'], '2020': df2020['change'] })

from matplotlib.pyplot import figure
figure(figsize=(16, 8), dpi=80)

# multiple line plots
plt.plot( '# of Days', '2012', data=df1, marker='', color='green', linewidth=2)
plt.plot( '# of Days', '2016', data=df1, marker='', color='blue', linewidth=2)
plt.plot( '# of Days', '2020', data=df1, marker='', color='red', linewidth=2)
# show legend
plt.legend()

plt.title("BTC price % change since cycle Halving date")
plt.xlabel("# of Days since cycle Halving date")
plt.ylabel("BTC price % change")
# show graph
plt.show()

rec_high_value_in_2012_cycle = df[index2012:index2016]['Close'].max()
rec_high_date_in_2012_cycle = df[index2012:index2016]['Close'].idxmax()

rec_high_value_in_2016_cycle = df[index2016:index2020]['Close'].max()
rec_high_date_in_2016_cycle = df[index2016:index2020]['Close'].idxmax()

rec_high_value_in_2020_cycle = df[index2020:]['Close'].max()
rec_high_date_in_2020_cycle = df[index2020:]['Close'].idxmax()

cycle_2012_start_date_from_previous_ath = None
for i in range(0, index2016):
    if df.iloc[i]['Close'] > rec_high_value_before_2012_halving:
        cycle_2012_start_index_from_previous_ath = i
        cycle_2012_cyclebase = df.iloc[i]['Close']
        break
    
cycle_2012_start_date_from_previous_ath = df.iloc[cycle_2012_start_index_from_previous_ath].name

df2012_ath = df[cycle_2012_start_date_from_previous_ath:halving2016].copy()

#df2012_ath['cyclebase'] = rec_high_value_before_2012_halving
df2012_ath['cyclebase'] = cycle_2012_cyclebase

df2012_ath['change'] = df2012_ath['Close'] / df2012_ath['cyclebase']

df2012_ath = df2012_ath.reset_index()

cycle_2016_start_date_from_previous_ath = None
for i in range(index2016, index2020):
    if df.iloc[i]['Close'] > rec_high_value_in_2012_cycle:
        cycle_2016_start_index_from_previous_ath = i
        cycle_2016_cyclebase = df.iloc[i]['Close']
        break
    
cycle_2016_start_date_from_previous_ath = df.iloc[cycle_2016_start_index_from_previous_ath].name

df2016_ath = df[cycle_2016_start_date_from_previous_ath:halving2020].copy()

#df2016_ath['cyclebase'] = rec_high_value_in_2012_cycle
df2016_ath['cyclebase'] = cycle_2016_cyclebase
df2016_ath['change'] = df2016_ath['Close'] / df2016_ath['cyclebase']

df2016_ath = df2016_ath.reset_index()

cycle_2020_start_date_from_previous_ath = None
for i in range(index2020, df.shape[0]):
    if df.iloc[i]['Close'] > rec_high_value_in_2016_cycle:
        cycle_2020_start_index_from_previous_ath = i
        cycle_2020_cyclebase = df.iloc[i]['Close']
        break
    
cycle_2020_start_date_from_previous_ath = df.iloc[cycle_2020_start_index_from_previous_ath].name


df2020_ath = df[cycle_2020_start_date_from_previous_ath:].copy()

#df2020_ath['cyclebase'] = rec_high_value_in_2016_cycle
df2020_ath['cyclebase'] = cycle_2020_cyclebase
df2020_ath['change'] = df2020_ath['Close'] / df2020_ath['cyclebase']

df2020_ath = df2020_ath.reset_index()

df2012_2016_avg_ath = df2016_ath.copy()
df2012_2016_avg_ath['change'] = (df2012_ath['change'] + df2016_ath['change']) / 2

x_max = max(df2012_ath.shape[0], df2016_ath.shape[0], df2020_ath.shape[0])

# Data
import matplotlib.pyplot as plt 
df1=pd.DataFrame({'# of Days': range(x_max), '2012': df2012_ath['change'], '2016': df2016_ath['change'], '2020': df2020_ath['change'], 'average(2012,2016)': df2012_2016_avg_ath['change'] })

from matplotlib.pyplot import figure
figure(figsize=(16, 8), dpi=80)

# multiple line plots
plt.plot( '# of Days', '2012', data=df1, marker='', color='green', linewidth=2)
plt.plot( '# of Days', '2016', data=df1, marker='', color='blue', linewidth=2)
plt.plot( '# of Days', '2020', data=df1, marker='', color='red', linewidth=2)
plt.plot( '# of Days', 'average(2012,2016)', data=df1, marker='', color='yellow', linewidth=2)
# show legend
plt.legend()

plt.title("BTC price % change since last record high in previous cycle")
plt.xlabel("# of Days")
plt.ylabel("BTC price % change")
# show graph
plt.show()


cycle_2020_start_value = df2020_ath['Close'][0]
df2012_ath['price'] = cycle_2020_start_value * df2012_ath['change']

df2016_ath['price'] = cycle_2020_start_value * df2016_ath['change']

df2020_ath['price'] = cycle_2020_start_value * df2020_ath['change']

df2012_2016_avg_ath['price'] = cycle_2020_start_value * df2012_2016_avg_ath['change']

x_max = max(df2012_ath.shape[0], df2016_ath.shape[0], df2020_ath.shape[0])
x_max = 500

x = pd.date_range(cycle_2020_start_date_from_previous_ath, cycle_2020_start_date_from_previous_ath + timedelta(x_max-1), freq='D')


x_today = df2020_ath.shape[0]
x_today = cycle_2020_start_date_from_previous_ath + timedelta(x_today-1)
y_today = df2020_ath.iloc[-1]['price']

y1 = est_max_value_using_2012 = df2012_ath['price'].max()
x1 = est_max_index_using_2012 = df2012_ath['price'].idxmax()
x1 = cycle_2020_start_date_from_previous_ath + timedelta(x1-1)

y2 = est_max_value_using_2016 = df2016_ath['price'].max()
x2 = est_max_index_using_2016 = df2016_ath['price'].idxmax()
x2 = cycle_2020_start_date_from_previous_ath + timedelta(x2-1)

y3 = est_max_value_using_average_2012_2016 = df2012_2016_avg_ath['price'].max()
x3 = est_max_index_using_average_2012_2016 = df2012_2016_avg_ath['price'].idxmax()
x3 = cycle_2020_start_date_from_previous_ath + timedelta(x3-1)

x0 = cycle_2020_start_date_from_previous_ath
y0 = cycle_2020_cyclebase

# Data
import matplotlib.pyplot as plt 
#df1=pd.DataFrame({'x': x, '2012': df2012_ath['price'], '2016': df2016_ath['price'], '2020': df2020_ath['price'], 'start value': cycle_2020_start_value })
#df1=pd.DataFrame({'x': x, '2012': df2012_ath['price'], '2016': df2016_ath['price'], '2020': df2020_ath['price'], 'average(2012,2016)': df2012_2016_avg_ath['price']})
df1=pd.DataFrame({'x': x, '2012': df2012_ath.iloc[0:x_max]['price'], '2016': df2016_ath.iloc[0:x_max]['price'], '2020': df2020_ath.iloc[0:x_max]['price'], 'average(2012,2016)': df2012_2016_avg_ath.iloc[0:x_max]['price']})

from matplotlib.pyplot import figure
figure(figsize=(16, 8), dpi=80)

# multiple line plots
plt.plot( 'x', '2012', data=df1, marker='', color='green', linewidth=2)
plt.plot( 'x', '2016', data=df1, marker='', color='blue', linewidth=2)
plt.plot( 'x', '2020', data=df1, marker='', color='red', linewidth=2)
plt.plot( 'x', 'average(2012,2016)', data=df1, marker='', color='yellow', linewidth=2)
#plt.plot( 'x', 'start value', data=df1, marker='', color='black', linewidth=2)

#plt.text(halving2020, cycle_2020_start_value, cycle_2020_start_value)
plt.plot(x0,y0,'ro') 
plt.plot(x1,y1,'ro') 
plt.plot(x2,y2,'ro') 
plt.plot(x3,y3,'ro') 
plt.plot(x_today,y_today,'ro') 
plt.text(x0, 0, (x0.strftime("%d %b %Y"),'${:,.0f}'.format(y0)))
plt.text(x1, y1, (x1.strftime("%d %b %Y"),'${:,.0f}'.format(y1)))
plt.text(x2, y2, (x2.strftime("%d %b %Y"),'${:,.0f}'.format(y2)))
plt.text(x3, y3, (x3.strftime("%d %b %Y"),'${:,.0f}'.format(y3)))
plt.text(x_today, 0, (x_today.strftime("%d %b %Y"),'${:,.0f}'.format(y_today)))

# show legend
plt.legend()

plt.title("BTC price projection")
plt.xlabel("Date")
plt.ylabel("BTC price")
# show graph
plt.show()

