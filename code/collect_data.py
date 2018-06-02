#!/usr/bin/env python
from bittrex import bittrex
import numpy as np
import datetime
import time
import csv
import matplotlib.pyplot as plt
from numpy import genfromtxt

def get_market_data (market):
    btc_data = np.zeros(4)
    btcsummary = api.getmarketsummary(market)
    while len(btcsummary)<1:
        btcsummary = api.getmarketsummary(market)
    btc_data[0] = btcsummary[0]['Last']
    btc_data[1] = btcsummary[0]['Volume']
    btc_data[2] = btcsummary[0]['Bid']
    btc_data[3] = btcsummary[0]['Ask']
#    print (btc_data[0,0])
#    print (btc_data[1,0])
    return btc_data
    pass 

def get_history (market):
    buy = 0
    sell = 0
    buy_quantity = 0
    sell_quantity = 0
    history = api.getmarkethistory(market)
    length = np.shape(history)[0]
    s = history[0]['TimeStamp']
    H,M,S = s.split(":")
    last_order = float(M)*60 + float(S)
    s = history[length-1]['TimeStamp']
    H,M,S = s.split(":")
    first_order = float(M)*60 + float(S)
    delta_time = last_order - first_order
    run_time = time.time()
    if delta_time < 0:
        delta_time += 60*60
    for i in range (0,length):
        if history[i]['OrderType'] == 'SELL':
            sell +=1
            sell_quantity += history[i]['Quantity']
        else:
            buy +=1
            buy_quantity += history[i]['Quantity']
    history_log = open('history_log.txt','a')
    s = repr(run_time) + 'Sell cnt: ' + repr(sell) + ' Sell Quantity: ' + repr(sell_quantity) + ' Buy cnt: ' + repr(buy) + ' Buy Quantity: ' + repr(buy_quantity) + ' time_span: ' + repr(delta_time) + '\n'
    history_log.write(s)
        
def order_book_data (market):
    sell = api.getorderbook(market,'sell')
    buy = api.getorderbook(market,'buy')
    #size = np.maximum(np.shape(sell)[0],np.shape(buy)[0])
    order = np.zeros((500,6))
    for i in range (0,500):
#        x = sell[i]
        order[i,0] = sell[i]['Quantity']
        order[i,2] = sell[i]['Rate']
        order[i,3] = buy[i]['Quantity']
        order[i,5] = buy[i]['Rate']
        if i>0:
            order[i,1] = order[i-1,1]+order[i,0]
            order[i,4] = order[i-1,4]+order[i,3]
        else:
            order[i,1] = order[i,0]
            order[i,4] = order[i,3]
#    plt.plot(order[:,2],order[:,1])
#    plt.plot(order[:,5],order[:,4])
#    plt.show()
    return order        




if __name__ == '__main__':
    key = ''
    secret = ''
    api = bittrex(key, secret)
# Market to trade at
    trade = 'USDT'
    currency = 'BTC'
    market = '{0}-{1}'.format(trade, currency)
    amount = 0.001
# if need init and 2 npy dont exist
#    btc_data_log = np.zeros((1,4))
#    btc_data_log[0,:] = get_market_data(market)
#    orders_log = np.zeros((1,500,6))
#    orders_log[0,:,:] = order_book_data(market)
#    np.save('order_log',orders_log)
#    np.save('btc_data_log',btc_data_log)
#    history_log = open('history_log.txt','w')
#    history_log.close()
# if files exist

    wd_ind = 0
    wd = open('wd.txt','r')
    wd_prev_val = wd.readline(1)
    wd.close()
    run_list = open('run_list.txt','r')
    for lines in run_list:
        my_run_ind = lines # will hold the last number
    my_run_ind = int(my_run_ind) + 1
    run_list.close()
    run_list = open('run_list.txt','a')
    s = repr(my_run_ind) + '\n'
    run_list.write(s)
    run_list.close()
    print ('run ind',my_run_ind)
    btc_data = np.zeros((1,4))
    orders = np.zeros((1,500,6))
    while 1 :
        time.sleep(300)
        wd = open('wd.txt','r')
        wd_val = wd.readline()
        run_ind = wd.readline()
        wd.close()
        if wd_val == wd_prev_val and int(run_ind) == my_run_ind-1:
            orders_log = np.load('order_log.npy')
            btc_data_log = np.load('btc_data_log.npy')
            print ('wd_val', wd_val)
            print ('wd_prev_val', wd_prev_val)
            print ('run_ind ',run_ind)
            print ('my_run_ind ',my_run_ind)
            print (btc_data_log)
            print (np.shape(btc_data_log))
            prev_time = time.time()
            break
        else:
            wd_prev_val = wd_val
            print ('wd alive')

    while 1 :
        loop_start_time_sec = time.time()
        if loop_start_time_sec - prev_time > 250: # protection in case loop took long time than abort. to prevent scenario when another agent has kicked off already
            exit()
        wd_ind += 1
        wd = open('wd.txt','w')
        s = repr(wd_ind) + '\n'
        wd.write(s)
        s = repr(my_run_ind) + '\n'
        wd.write(s)
        wd.close()        
        print (loop_start_time_sec)
        btc_data[0,:] = get_market_data(market)
        get_history(market)
        orders[0,:,:] = order_book_data(market)
        btc_data_log = np.append(btc_data_log,btc_data,axis=0)
        orders_log = np.append(orders_log,orders,axis=0)
        np.save('order_log',orders_log)
        np.save('btc_data_log',btc_data_log)
        time.sleep (60)
        prev_time = time.time()
        


#    btc_balance = get_balance(currency)
#    usd_balance = get_balance(trade)
#    btc_summary = api.getmarketsummary(market)
#    print (btc_summary)
#    print ()

#    print (np.shape(history)[0])
#    print (np.shape(order)[0])
#    history = api.getmarketsummaries()
#    print (btc_balance)
#    print (usd_balance)
#    print (np.shape(history))
#    print (history[1:3])
#    x = api.selllimit(market, amount, btcprice+1000)
#    x = api.selllimit(market, amount, btcprice+1000)
#    x = api.buylimit(market, amount, btcprice-1000)
#    x = api.buylimit(market, amount, btcprice-1000)
#    time.sleep(10)
#    y = api.getopenorders(market)
#    while len(y)>0:
#        uuid = y[0]['OrderUuid']
#        api.cancel(uuid)
#        print (uuid)
#        y = api.getopenorders(market)
#    time.sleep(5)
#    btcbalance = api.getbalance(currency)
#    print ('Balance = ',btcbalance)
#    usdbalance = api.getbalance(trade)
#    print ('usd Balance = ',usdbalance)
