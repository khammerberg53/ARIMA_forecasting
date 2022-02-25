
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning, HessianInversionWarning, ConvergenceWarning
import warnings

#in practice do not supress these warnings, they carry important information about the status of your model
warnings.filterwarnings('ignore', category=ValueWarning)
warnings.filterwarnings('ignore', category=HessianInversionWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# select your stock's ticker symbol
tickerSymbol = 'SPY'
data = yf.Ticker(tickerSymbol)

# converting price data to returns 
prices = data.history(start='2017-01-01', end='2022-01-01').Close
returns = prices.pct_change().dropna()

# function for calculating standard deviation 
def std_dev(data):
    # Get number of observations
    n = len(data)
    # Calculate mean
    mean = sum(data) / n
    # Calculate deviations from the mean
    deviations = sum([(x - mean)**2 for x in data])
    # Calculate Variance & Standard Deviation
    variance = deviations / (n - 1)
    s = variance**(1/2)
    return s

# function for calculating sharpe ratio
def sharpe_ratio(data, risk_free_rate=0):
    # Calculate Average Daily Return
    mean_daily_return = sum(data) / len(data)
    print(f"mean daily return = {mean_daily_return}")
    # Calculate Standard Deviation
    s = std_dev(data)
    # Calculate Daily Sharpe Ratio
    daily_sharpe_ratio = (mean_daily_return - risk_free_rate) / s
    # Annualize Daily Sharpe Ratio
    sharpe_ratio = 252**(1/2) * daily_sharpe_ratio
    return sharpe_ratio

def run_simulation(returns, prices, amt, order, thresh, verbose=True, plot=True):
    if type(order) == float:
        thresh = None
        
    curr_holding = False
    sum_list = []
    events_list = []
    sharpe_list = []
    init_amt = amt

    #go through dates
    for date, r in tqdm (returns.iloc[14:].items(), total=len(returns.iloc[14:])):
        
        

        #get data til just before current date
        curr_data = returns[:date]
        
        # check if using ARIMA from order
        if type(order) == tuple:
                #fit model
                model = ARIMA(curr_data, order=order).fit()

                #get forecast
                pred = model.forecast()
                print(pred)
                float_pred = float(pred)



        #if you predict a high enough return and not holding, buy stock
        # order for random strat and tuple for ARIMA
        if float_pred > thresh \
         or (order == 'last' and curr_data[-1] > 0):
        
            
           
            buy_price = prices.loc[date]
            events_list.append(('b', date))
            int_buy_price = int(buy_price)
            sum_list.append(int_buy_price)
            curr_holding = True
            if verbose:
                print('Bought at $%s'%buy_price)
                print('Predicted Return: %s'%round(pred,4))
                print(f"Current holdings = {sum(sum_list)}")
                print('=======================================')
            continue

        #if you predict below the threshold return, sell the stock
        if (curr_holding) and \
        ((type(order) == float and np.random.random() < order) 
         or (type(order) == tuple and float_pred < thresh)
         or (order == 'last' and curr_data[-1] > 0)):
        
            
            sell_price = prices.loc[date]
            
            total_return = len(sum_list) * sell_price 

            ret = (total_return-sum(sum_list))/sum(sum_list)
            amt *= (1+ret)
            events_list.append(('s', date, ret))
            sharpe_list.append(ret)
            sum_list.clear()
            curr_holding = False

            if verbose:
                print('Sold at $%s'%sell_price)
                print('Predicted Return: %s'%round(pred,4))
                print('Actual Return: %s'%(round(ret, 4)))
                print('=======================================')
            
            

                
    if verbose:
        sharpe = sharpe_ratio(sharpe_list, risk_free_rate=0)
        print('Total Amount: $%s'%round(amt,2))
        print(f"Sharpe Ratio: {sharpe}")
        
    #graph
    if plot:
    
        plt.figure(figsize=(10,4))
        plt.plot(prices[14:])

        y_lims = (int(prices.min()*.95), int(prices.max()*1.05))
        shaded_y_lims = int(prices.min()*.5), int(prices.max()*1.5)

        for idx, event in enumerate(events_list):
            plt.axvline(event[1], color='k', linestyle='--', alpha=0.4)
            if event[0] == 's':
                color = 'green' if event[2] > 0 else 'red'
                plt.fill_betweenx(range(*shaded_y_lims), 
                                  event[1], events_list[idx-1][1], color=color, alpha=0.1)

        tot_return = round(100*(amt / init_amt - 1), 2)
        sharpe = sharpe_ratio(sharpe_list, risk_free_rate=0)
        tot_return = str(tot_return) + '%'
        plt.title("%s Price Data\nThresh=%s\nTotal Amt: $%s\nTotal Return: %s"%(tickerSymbol, thresh, round(amt,2), tot_return), fontsize=20)
        plt.ylim(*y_lims)
        plt.show()
        print(sharpe)
    
    return amt

# A model with a dth difference to fit and ARMA(p,q) model is called an ARIMA process 
# of order (p,d,q). You can select p,d, and q with a wide range of methods, 
# including AIC, BIC, and empirical autocorrelations (Petris, 2009).


for thresh in [0.001]:
    run_simulation(returns, prices, 100000, (7,0,0), thresh, verbose=True)
