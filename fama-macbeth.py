# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import date

# Import portfolio returns for the 25 year period (need 5 years of additional data for the rolling window)
returns = pd.read_csv("portfolio_returns_VW.csv", index_col='Date', na_values=[-999, -99.99])
returns.index = pd.to_datetime(returns.index, format='%Y%m')
returns = returns.loc[date(1995, 1, 1):date(2020, 10, 1)]

# Import factor values for the 25 year period
factors = pd.read_csv("three_factors.csv", index_col='Date')
factors.index = pd.to_datetime(factors.index, format='%Y%m')
factors = factors.loc[date(1995, 1, 1):date(2020, 10, 1)]

# Import unemployment rate data and merge with existing 'factors' DataFrame
unemp = pd.read_csv("UNRATE.csv", index_col='DATE')
unemp.index.rename('Date', inplace=True)
unemp.index = pd.to_datetime(unemp.index, format='%Y-%m-%d')
factors = factors.merge(unemp, how='inner', on='Date')

# Calculate excess returns using the 'RF' column in the 'factors' DataFrame
excess_returns = returns.sub(factors['RF'], axis=0)

# Create DataFrames for storing the factor coefficients
rolling_beta = pd.DataFrame(index=excess_returns.index, columns=excess_returns.columns)
rolling_SMB = pd.DataFrame(index=excess_returns.index, columns=excess_returns.columns)
rolling_HML = pd.DataFrame(index=excess_returns.index, columns=excess_returns.columns)
rolling_unemp = pd.DataFrame(index=excess_returns.index, columns=excess_returns.columns)

# Using time-series regression, obtain the factor coefficients for each portfolio at each point in time
# At each point in time, the previous 60 months of data is used
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

exog = factors[['Mkt-RF', 'SMB', 'HML', 'UNRATE']]
exog = sm.add_constant(exog)
for label, content in excess_returns.iteritems():
    endog = content
    rols = RollingOLS(endog, exog, window=60)
    rres = rols.fit(cov_type='HAC',cov_kwds={'maxlags':1})
    rolling_beta[label] = rres.params['Mkt-RF']
    rolling_SMB[label] = rres.params['SMB']
    rolling_HML[label] = rres.params['HML']
    rolling_unemp[label] = rres.params['UNRATE']

# CAPM

# At each point in time, run a cross-sectional regression
from sklearn.linear_model import LinearRegression

alpha = []
gamma1 = []
reg = LinearRegression()
for index, beta in rolling_beta.iloc[60:].iterrows():
    df = pd.DataFrame({'Beta': beta})
    X = np.array(df).reshape(-1, 1)
    y = excess_returns.loc[index] 
    reg.fit(X, y)
    gamma1.append(reg.coef_[0])
    alpha.append(reg.intercept_)

# Calculate t-statistics
alpha_mean = np.mean(np.array(alpha))
gamma1_mean = np.mean(np.array(gamma1))
alpha_std = np.std(np.array(alpha))
gamma1_std = np.std(np.array(gamma1))
T = len(rolling_beta.iloc[60:])
alpha_t = alpha_mean * math.sqrt(T) / alpha_std
gamma1_t = gamma1_mean * math.sqrt(T) / gamma1_std
print("CAPM Test Statistics:\n")
print(f"Pricing error: {alpha_t:.4f}")
print(f"Market excess return: {gamma1_t:.4f}")

# Three-Factor

alpha = []
gamma1 = []
gamma2 = []
gamma3 = []
reg = LinearRegression()
for index, beta in rolling_beta.iloc[60:].iterrows():
    smb = rolling_SMB.loc[index] 
    hml = rolling_HML.loc[index] 
    df = pd.DataFrame({'Beta': beta, 'SMB': smb, 'HML': hml})
    X = np.array(df).reshape(-1, 3)
    y = excess_returns.loc[index] 
    reg.fit(X, y)
    gamma1.append(reg.coef_[0])
    gamma2.append(reg.coef_[1])
    gamma3.append(reg.coef_[2])
    alpha.append(reg.intercept_)

alpha_mean = np.mean(np.array(alpha))
gamma1_mean = np.mean(np.array(gamma1))
gamma2_mean = np.mean(np.array(gamma2))
gamma3_mean = np.mean(np.array(gamma3))
alpha_std = np.std(np.array(alpha))
gamma1_std = np.std(np.array(gamma1))
gamma2_std = np.std(np.array(gamma2))
gamma3_std = np.std(np.array(gamma3))
T = len(rolling_beta.iloc[24:])
alpha_t = alpha_mean * math.sqrt(T) / alpha_std
gamma1_t = gamma1_mean * math.sqrt(T) / gamma1_std
gamma2_t = gamma2_mean * math.sqrt(T) / gamma2_std
gamma3_t = gamma3_mean * math.sqrt(T) / gamma3_std
print("Three-Factor Test Statistics:\n")
print(f"Pricing error: {alpha_t:.4f}")
print(f"Market excess return: {gamma1_t:.4f}")
print(f"SMB: {gamma2_t:.4f}")
print(f"HML: {gamma3_t:.4f}")

# Three Factor + Unemployment

alpha = []
gamma1 = []
gamma2 = []
gamma3 = []
gamma4 = []
reg = LinearRegression()
for index, beta in rolling_beta.iloc[60:].iterrows():
    smb = rolling_SMB.loc[index] 
    hml = rolling_HML.loc[index] 
    unrate = rolling_unemp.loc[index] 
    df = pd.DataFrame({'Beta': beta, 'SMB': smb, 'HML': hml, 'UNRATE': unrate})
    X = np.array(df).reshape(-1, 4)
    y = excess_returns.loc[index] 
    reg.fit(X, y)
    gamma1.append(reg.coef_[0])
    gamma2.append(reg.coef_[1])
    gamma3.append(reg.coef_[2])
    gamma4.append(reg.coef_[3])
    alpha.append(reg.intercept_)

alpha_mean = np.mean(np.array(alpha))
gamma1_mean = np.mean(np.array(gamma1))
gamma2_mean = np.mean(np.array(gamma2))
gamma3_mean = np.mean(np.array(gamma3))
gamma4_mean = np.mean(np.array(gamma4))
alpha_std = np.std(np.array(alpha))
gamma1_std = np.std(np.array(gamma1))
gamma2_std = np.std(np.array(gamma2))
gamma3_std = np.std(np.array(gamma3))
gamma4_std = np.std(np.array(gamma4))
T = len(rolling_beta.iloc[24:])
alpha_t = alpha_mean * math.sqrt(T) / alpha_std
gamma1_t = gamma1_mean * math.sqrt(T) / gamma1_std
gamma2_t = gamma2_mean * math.sqrt(T) / gamma2_std
gamma3_t = gamma3_mean * math.sqrt(T) / gamma3_std
gamma4_t = gamma4_mean * math.sqrt(T) / gamma4_std
print("Three Factor+Unemployment Test Statistics:\n")
print(f"Pricing error: {alpha_t:.4f}")
print(f"Market excess return: {gamma1_t:.4f}")
print(f"SMB: {gamma2_t:.4f}")
print(f"HML: {gamma3_t:.4f}")
print(f"UNRATE: {gamma4_t:.4f}")