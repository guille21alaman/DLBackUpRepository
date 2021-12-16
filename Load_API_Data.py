###############
# DESCRIPTION #
###############

"""
    In this scrpit, we provide a function for querying, saving and loading the necessary data from AlphaVantage API.
    The necessary libraries to use the functions in this script are shown right below this lines.
"""

import time
import numpy as np
import pandas as pd
import os 


#######################
# QUERY AND SAVE DATA #
#######################

"""
    Description of the function: Querying data from AlphaVantage API 
    Inputs:
        * symbols: stock symbols to be extracted (i.e., Apple: APPL). Reference list of S&P 500 symbols: 
        https://en.wikipedia.org/wiki/List_of_S%26P_500_companies 
        * ts: timeSeries object from AlphaVantage to query historical prices
    
    Comments: This function is meant to be called only once when actually querying data from the API.
    Once the stock with which one wants to work has been stored in the folder "data", there is no need 
    to call the saveData() function anymore
"""

def saveData(symbols, ts):

    #information print
    print("\nQuerying data from AlphaVantage API...")
    
    #needed to avoid more than 5 queries per minute (limit of AlphaVantage API) - only for the case of more than 5 symbols
    count = 1

    #loop until every desired stock prices have been query
    while count <= len(symbols):

        #loop over the stocks in the symbols list
        for stock in symbols:
            #information print
            print("Current stock processed: ", stock)

            #store the pandas dataframe with historical prices in the data folder with the name of the stock symbol as name
            ts.get_daily_adjusted(stock, "full")[0].to_csv("data/%s.csv" %(stock), sep=";", decimal=",")

            #increase count for the track of query limit per minute
            count += 1

            #if the count is divisible by 5, then already 5 queries in this minute, sleep for a minute and continue
            # only if there are more than 5 symbols
            if count % 5 == 0:
                print("Sleeping... The API should rest a bit :)")
                time.sleep(60+1)
    
    print("-----------------------")
    print("\n\n\n\n\n\n\n")


#############
# LOAD DATA #
#############

"""
    Description of the function: function to load the csv file of each stock (created by the loadData function)
    into a pandas dataframe. Minor changes to the column names and index are also performed.

    Inputs:
        * directory: folder from which the data should be loaded
        (note that this directory should contain only csv files as imported from AlphaVantage API by function loadData)
        * columns: list with the new desired names of the columns (rename)

    Return: dictionary with one pd.dataframe per stock in which the key is the file name as stored in the
    directory and the values are the dataframes
"""


def loadData(directory, columns):

    #dictionary to the dataframes
    raw_data = {}

    #loop over the files in the directory where data is stored
    for file in os.listdir(directory):

        #information print
        print("Current stock processed: ", file)

        #read file (of an stock)
        raw_data[file] = pd.read_csv(directory+"/"+file, sep=";", decimal=",")

        #rename columns (using columns list taken as an argument)
        #hard-coded because the output data from AlphaVantage Follows always the same structure
        raw_data[file] = raw_data[file].rename(columns={
            "1. open": columns[0],
            "2. high": columns[1],
            "3. low": columns[2],
            "4. close": columns[3],
            "5. adjusted close": columns[4],
            "6. volume": columns[5],
            "7. dividend amount": columns[6],
            "8. split coefficient": columns[7]
        })

        #make date index
        raw_data[file] = raw_data[file].set_index("date")
        #convert to datetime (double chek)
        raw_data[file].index = pd.to_datetime(raw_data[file].index)

        #sort values by dates (ascending)
        raw_data[file] = raw_data[file].sort_values(by="date", ascending = True)

    #return dictionary 
    return raw_data