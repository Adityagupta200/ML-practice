import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlalchemy
import sqlalchemy as SQLal
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import AbstractHolidayCalendar, nearest_workday, Holiday
from pytz import all_timezones
df = pd.read_csv('C:\\Users\\abc\\Downloads\\NewDelhi.csv',parse_dates=['datetime'], index_col='datetime')
# print(df) --> prints the data in tabular form.
# print(df['tempmin'].max()) --> prints the maximum value in a coloumn.
# df2 = df['datetime'][df['tempmin']>=20] --> checks the condition in the square bracket and prints the contents in the mentioned coloumn.
# df.fillna(0,inplace=True) --> replaces the or cell containing nothing with 0.
# rows, columns = df.shape --> returns a tuple containing the rows and columns like (rows, columns).
# print(rows)
# print(df.head(5)) --> returns the dataframe containing first 5 rows.
# print(df.tail(1)) --> prints the last row of the dataframe 'df'.
# print(df[2:5]) --> to print rows from 2 to 4 (5 is not included).
# print(df[:])--> to print all rows.
# print(df.columns) --> prints all columns.
# print(type(df['icon'])) --> prints type of contents in the 'icon' column.
# print(df[['icon','datetime','tempmax']]) --> prints the mentioned columns.
# print(df.describe()) --> returns dataframe containing some statistical results of the columns containing elements of type float or int.
# print(df.index) --> returns the range containing the start and ending index.
# df.set_index('icon',inplace=True)
# print(df.loc('rain')) --> prints the particular specified row.
# df2 = pd.read_excel('excel file link', 'sheet name') --> creates a data frame after reading the excel file.
# A python dictionary or a tuple list can also be used to create a data frame.
# Syntax using with a python dictionary:-
weather_data = {
    'day' : ['1/1/2017', '1/2/2017', '1/3/2017'],
    'temprature' : [32,35,28],
    'windspeed' : [6,7,12],
    'event' : ['Rain', 'Snow', 'Sunny']
}

# print(pd.DataFrame(weather_data))
# Syntax using a tuple list:-
list = [
    ('1/1/2017',32,6,'Rain'),
    ('1/2/2017',35,7,'Sunny'),
    ('1/3/2017',28,2,'Snow')
]
# print(pd.DataFrame(list, columns=["date","temprature","windspeed","event"]))
weather_data_list_of_dictionary = [
    {'date':'1/1/2017','temprature':32, 'windspeed':6, 'event' : 'Rain'},
    {'date':'1/2/2017','temprature':35, 'windspeed':7, 'event' : 'Sunny'},
    {'date':'1/3/2017','temprature':28, 'windspeed':2, 'event' : 'Snow'}
]
# print(pd.DataFrame(weather_data_list_of_dictionary))
# df2 = pd.read_excel('C:\\Users\\abc\\Downloads\\stock_data.xlsx',skiprows=1) --> skipped 1st row
# df2 = pd.read_excel('C:\\Users\\abc\\Downloads\\stock_data.xlsx',header=None,names=["ticker", "eps", "revenue", "price", "people"])
# df2 = pd.read_excel('C:\\Users\\abc\\Downloads\\stock_data.xlsx', na_values={  #--> converts the given values
#     'eps' : ['not available', 'n.a.'],
#     'revenue' : [-1],
#     'people' : ['ratan tata', 'n.a.']
# })
# df2.to_csv('new_csv', index=False, columns=['revenue','people'], header= False) --> creates a new csv file in which the index column is not present and only 'revenue' and 'people' columns are present, and header is absent.
def na_converter(cell):
    if cell == "NA" :
        return 'sam walton'
    return cell
# df2 = pd.read_excel('C:\\Users\\abc\\Downloads\\stock_data.xlsx',converters={ #--> using the method na_converter.
#     'people' : na_converter
# })
# df2.to_excel('new_excel.xlsx',sheet_name = 'stocks')

df_stocks = pd.DataFrame({
    'tickers' : ['google', 'RJL', 'MSFT'],
    'price' : [845, 65, 64],
    'eps' : [27.82, 4.61, 2.12]
})
df_weather = pd.DataFrame({
    'date' : ['1/1/2017', '1/2/2017', '1/3/2017','1/4/2017','1/5/2017','1/6/2017','1/7/2017'],
    'windspeed' : [6,7,-99999,7,-99999,2,5],
    'temprature' : [32,-99999,28,-99999,32,31,34],
    'event' : ['Rain', 'Sunny', 'Snow', 'No Event', 'Rain', 'Sunny', 'No Event']
})
# df_weather.to_csv('weather_data.csv')
# with pd.ExcelWriter('stocks_weather.xlsx') as writer : --> creates an excel file named 'stocks_weather.xlsx' containing to sheets named 'stocks' and 'weather'.
#     df_stocks.to_excel(writer, sheet_name='stocks')
#     df_weather.to_excel(writer,sheet_name='weather')
# For more properties of pandas library surf pandas documentation.

df3 = pd.read_csv('weather_data.csv', parse_dates= ['date']) # parse_dates = ['date'] changed the type of elements in the date column.
# print(type(df3.date[0]))
# df3.set_index('date',inplace=True)
# del df3['Unnamed: 0'] --> For deletion of a column.
# new_df = df.fillna(0) --> Fills all cells of value NaN with 0.
# new_df = df.fillna({
#     'temprature' : 0,
#     'windspeed' : 0,
#     'stations' : 'no event'
# })
# df4 = df.fillna(method = 'ffill') --> fills the value of the previous cell which is not null similarly in place of 'ffill' 'bfill' fills the value of the cell with the same value of the next cell which is not null.

# new_df = df_weather.replace({
#     'temprature' : -99999,
#     'windspeed' : -99999,
#     'event' : 'No Event'
# }, np.NAN)
new_df = df_weather.replace({
    -99999 : np.NAN,
    'No Event' : 'Nothing'
})
# print(df)
g = df.groupby('icon') # groups the data on the basis of the column provided.
# for city, city_df in g:
#     print(city)
#     print(city_df)
# print(g.get_group('clear-day')) --> returns the specified group.
# print(g.max('windspeed')) --> prints maximum values in the 'windspeed' column with the groups
# print(g.describe())
# g.plot()
# plt.show()
india_weather = pd.DataFrame({
    "city" : ['delhi','banglore', 'pune'],
    "temprature" : [32,35,45],
    "humidity" : [80,70,60]
},index=[0,1,2])
us_weather = pd.DataFrame({
    "city" : ['brooklyn', 'orlando', 'New York'],
    "temprature" : [35,21,78],
    "humidity" : [34,13,89]
},index=[2,1,0])
# merged_weather = pd.concat([india_weather,us_weather], ignore_index=True) --> concats the two DataFrames into one.
merged_weather = pd.concat([india_weather,us_weather],axis=1)
# print(merged_weather)
# print(merged_weather.loc['india']) --> returns a subset of the merged_weather dataframe.
s = pd.Series(["Rain","Sunny","Cloudy"], name="Event")
df4 = pd.concat([merged_weather,s],axis=1)
# print(df.columns)
# print(pd.crosstab([df.datetime, df.visibility],[df.icon,df.windspeed], margins=True, normalize='index'))
# print(pd.crosstab(df.datetime, df.icon, values=df.windspeed, aggfunc=np.average,margins=True))
# print(df.columns)
# print(df.loc['2023-03-24'].T)
# df.windspeed.resample('Q').mean().plot(kind= "bar")
# df.windspeed.plot();
# plt.show()
df5 = pd.read_csv('C:\\Users\\abc\Documents\\stocks.csv',parse_dates=['Date'])
df5_noDates = df5.drop('Date',axis=1)
# rng = pd.date_range(start="6/1/2017",end="6/30/2017",freq='B')
# df5_noDates.set_index(rng,inplace=True)
# print(df5_noDates["2017-06-01":"2017-06-20"].Close.mean())
# print(df5_noDates.asfreq('D',method='pad'))
# rng = pd.date_range(start="1/1/2023", periods=10, freq='H')
# print(pd.Series(np.random.randint(1,10,len(rng)),index=rng))
usb = CustomBusinessDay(calendar = USFederalHolidayCalendar())
rng = pd.date_range(start="1/1/2023", end="2/22/2023",freq = usb)
# df5_noDates.set_index(rng,inplace=True)
# print(df5_noDates)
class myBirthdayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("Aditya's birthday",month=1, day=14, observance=nearest_workday) # If a holiday is on a national holiday(according to US calendar) observance = nearest_workday finds the nearest working day and sets it to be a holiday
    ]
myCalendar = CustomBusinessDay(calendar=myBirthdayCalendar())
# print(pd.date_range(start="1/1/2023",end="1/30/2023",freq=myCalendar))
b = CustomBusinessDay(weekmask='Sun Mon Tue Wed Thu', holidays=['2023-03-29'])
# print(pd.date_range(start='3/1/2023', end= '3/30/2023', freq=b))
dates = ['2017-01-05 2:30:00 PM', 'Jan 5, 2017 14:30:00', '01/05/2017', '2017.01.05', '20170105']
# print(pd.to_datetime(dates))
# print(pd.to_datetime('5#1#2017', format='%d#%m#%Y'))
dates_invalid = ['abc']
# print(pd.to_datetime(dates_invalid, errors='ignore'))
# print(pd.to_datetime(dates_invalid, errors='coerce')) --> sets the invalid date-time to be 'NaT' and converts the rest.
t = 1501356749
dt = pd.to_datetime([t],unit='s')
# print(dt.view('int64'))
y = pd.Period('2016')
# print(y.start_time)
# print(y.end_time)
m = pd.Period('2016-1', freq='M')
# print(m.start_time)
# print(m.end_time)
h = pd.Period('2016-01-23 23:00:00', freq='H')
# print(h+1)
q = pd.Period('2017Q1', freq='Q-JAN')
# print(q)
q2 = pd.Period('2018Q2', freq='Q-JAN') # freq='Q-JAN' means the quarter ending on january instead of december.
# print(q2)
# print(q2-q)
idx = pd.period_range('2011', '2017', freq='Q-JAN')
# print(idx)
idx2 = pd.period_range('2011', periods=10, freq='Q-JAN')
# print(idx2)
ps = pd.Series(np.random.randn(len(idx2)), idx2)
# print(ps)
# print(ps['2011' : '2013'])
ps2 = ps.to_timestamp()
# print(ps2)

df = pd.DataFrame({
    'Line Item' : ['Revenue', 'Expenses', 'Profit'],
    '2017Q1' : [115904, 86544, 29360],
    '2017Q2' : [120854, 89485, 31369],
    '2017Q3' : [118179, 87484,30695],
    '2017Q3' : [118179, 87484,30695],
    '2017Q4' : [118179, 87484,30695],
    '2018Q1' : [118179, 87484,30695]
})
df.set_index('Line Item', inplace=True)
df = df.T
df.index = pd.PeriodIndex(df.index, freq='Q-JAN')
df['start date'] = df.index.map(lambda x : x.start_time)
df6 = pd.DataFrame({
    'Date Time' : ['2017-08-17 09:00:00', '2017-08-17 09:15:00', '2017-08-17 09:30:00', '2017-08-17 10:00:00', '2017-08-17 10:30:00', '2017-08-17 11:00:00'],
    'Price' : ['72.38', '71.00', '71.67', '72.80', '73.00', '72.50']
})
df6.to_csv('Price.csv')
df7 = pd.read_csv('Price.csv', parse_dates=['Date Time'], index_col=False)
df7.set_index('Date Time', inplace=True)
# df7 = df7.tz_localize(tz='US/Eastern')
# print(all_timezones) --> To see all timezones.
df7.drop('Unnamed: 0', inplace=True, axis=1)
rng2 = pd.date_range(start="1/1/2017", periods=2, freq='H', tz="dateutil/Asia/Dili")
# rng3 = pd.date_range(start=)
rng3 = pd.date_range(start="2017-08-22 09:00:00", periods=10, freq='30min')
s = pd.Series(range(10), index= rng3)
m = s.tz_localize(tz='Asia/Calcutta')
d =s.tz_localize(tz='Asia/Dili')
# print(df7)
# print(df7.shift(1)) --> To shift the data of the previous day to the next day.
df7['difference'] = df7.shift(-1)['Price'] - df7.shift(1)['Price']
# print(df7)
df7.index = pd.date_range(start='2017-08-17 09:00:00', periods=6, freq='B')
print(df7.tshift(1))




