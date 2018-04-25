# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:43:13 2018

@author: averma
"""

# =============================================================================
# Linear Regression
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

# =============================================================================
# Simple Linear Regression
# =============================================================================

#y=ax+b

#create data 
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
plt.scatter(x, y);

#use Scikit-Learn's LinearRegression estimator to fit this data and construct the best-fit line:
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)

model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, 10, 1000) # lot of data, on a straight line so when we print, it look like a line
yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit);

#relevant result param
print("Model slope:    ", model.coef_[0])
print("Model intercept:", model.intercept_)

#for y=a0+a1x1+a2x2+
rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X, [1.5, -2., 1.])

model.fit(X, y)
print(model.intercept_)
print(model.coef_)

# =============================================================================
# #Basis Function Regression
# =============================================================================
#not y=a0+a1x1+a2x2+a3x3+
#rather this y=a0+a1x+a2x2+a3x3+⋯

#Polynomial basis functions

from sklearn.preprocessing import PolynomialFeatures
x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)
poly.fit_transform(x[:, None]) 
#transformer has converted our one-dimensional array into a three-dimensional array by taking the exponent of each value

#use a pipeline. Let's make a 7th-degree polynomial model in this way
from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())

#make data , or example, here is a sine wave with noise:
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)

poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit);

# =============================================================================
# Example: Predicting Bicycle Traffic
# =============================================================================
dataFilePatch="C:/JDeveloper/EclipseWS/PythonDataScienceHandbook/notebooks/data/"

import pandas as pd
counts = pd.read_csv(dataFilePatch+'FremontBridge.csv', index_col='Date', parse_dates=True)
weather = pd.read_csv(dataFilePatch+'BicycleWeather.csv', index_col='DATE', parse_dates=True)

#compute the total daily bicycle traffic, and put this in its own dataframe:
daily = counts.resample('d').sum()
daily['Total'] = daily.sum(axis=1)
daily = daily[['Total']] # remove other columns

#add days
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(7):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)

#we might expect riders to behave differently on holidays; let's add an indicator of this as well:
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012', '2016')
daily = daily.join(pd.Series(1, index=holidays, name='holiday'))
daily['holiday'].fillna(0, inplace=True)

#hours of daylight would affect how many people ride; 
#let's use the standard astronomical calculation to add this information
def hours_of_daylight(date, axis=23.44, latitude=47.61):
    """Compute the hours of daylight for the given date"""
    days = (date - pd.datetime(2000, 12, 21)).days
    m = (1. - np.tan(np.radians(latitude))
         * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
daily[['daylight_hrs']].plot()
plt.ylim(8, 17)

#We can also add the average temperature and total precipitation to the data.
#also, add a flag that indicates whether a day is dry (has zero precipitation):

# temperatures are in 1/10 deg C; convert to C
weather['TMIN'] /= 10
weather['TMAX'] /= 10
weather['Temp (C)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])

# precip is in 1/10 mm; convert to inches
weather['PRCP'] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(int)

daily = daily.join(weather[['PRCP', 'Temp (C)', 'dry day']])

#Let's add a counter that increases from day 1, and measures how many years have passed
daily['annual'] = (daily.index - daily.index[0]).days / 365

# Drop any rows with null values
daily.dropna(axis=0, how='any', inplace=True)

#run the linear regression
column_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday',
                'daylight_hrs', 'PRCP', 'dry day', 'Temp (C)', 'annual']
X = daily[column_names]
y = daily['Total']

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
daily['predicted'] = model.predict(X) # we should have used diffrent set for fit and predict

#Finally, we can compare the total and predicted bicycle traffic visually
daily[['Total', 'predicted']].plot(alpha=0.5);


#we can take a look at the coefficients of the linear model to estimate how much each feature contributes to the daily bicycle count
params = pd.Series(model.coef_, index=X.columns)
print(params)

#These numbers are difficult to interpret without some measure of their uncertainty. 
#We can compute these uncertainties quickly using bootstrap resamplings of the data
from sklearn.utils import resample
np.random.seed(1)
err = np.std([model.fit(*resample(X, y)).coef_
              for i in range(1000)], 0)

#With these errors estimated, let's again look at the results
print(pd.DataFrame({'effect': params.round(0),'error': err.round(0)}))

#now, We see that for each additional hour of daylight, 129 ± 9 more people choose to ride
#or, a temperature increase of one degree Celsius encourages 65 ± 4 people to grab their bicycle.
#each inch of precipitation means 665 ± 62 more people leave their bike at home. 
#Once all these effects are accounted for, we see a modest increase of 27 ± 18 new daily riders each year.

