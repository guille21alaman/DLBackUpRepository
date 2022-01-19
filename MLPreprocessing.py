###############
# DESCRIPTION #
###############

"""
    In this scrpit, functions for preprocessing raw data are defined:

    * Train and test split according to dates described in README.md
    * Inclusion of lagged prices
    * Inclusion of financial technical indicators as additional features
    * X and Y splits are carried out

    Imports needed in the functions described here are shown below
"""

import datetime
import talib
import pandas as pd



#################################
# PREPROCESS DATA FOR MODELLING #
#################################

"""
    Transformation from raw data to preprocessed (feautre engineering) - except for X and Y split

    Inputs:

    * raw_data: dictionary with raw data as described in Load_API_DATA
    * columns: columns to be kept for next steps from the original raw data (feature selection)
    * cutoffDate: starting data of training set period
    * train_end: end of train period
    * test_begin: beginning of validation period
    * test_end: end of the validation period (note that test period will only be queried at the very end of the project)
    * shiftDays: list of days to shift for the generation of targets (in the case of this project 1,5,10 as described in the README - 3 targets)
    * indicators: technical indicators to be extracted as additional features
    * intervals: intervals for stock price direction classification (2 as described in README)
    * lags: list of lagged days to be included as an additional features (i.e. if [1,2,5] the price of the previous 1,2 and 5 days are included as additional features)
    * target: price type (string like "adjustedClose")
"""


def preprocess(raw_data, columns, cutoffDate, train_end, test_begin, test_end, shiftDays, indicators, intervals, lags, target):
    
    #print information
    print("\n Time for some cool preprocessing...")
    
    data = {}
    train = {}
    test = {}



    for file in raw_data:

        print("Current stock processed: ", file)

        #select only columns to  be used later
        data[file] = raw_data[file][columns]


        #filter out dates
        #add 1 year more to compute metrics (i.e. Moving averages etc.) - will generate nulls - then again filter out dates after everything
        data[file] = data[file].loc[datetime.date(year=cutoffDate[2]-2, month=cutoffDate[1], day=cutoffDate[0]):datetime.date(year=test_end[2], month=test_end[1], day=test_end[0])]
        
        print("Data used", data[file].iloc[0].name)

        #calculate percentage increase of close price (target) - do it for an interval of days 
        for shiftDay in shiftDays:

            #append targets
            data[file]["target%sIncrease%sDays" %(target, shiftDay)] = ((data[file][target] - data[file][target].shift(shiftDay)) / data[file][target].shift(shiftDay))*100

            #and shift properly, we want to predict tomorrow's increase (or 5/10 day increase respectively). Not today! (originally mistaken and lead to a LOT of time loss) 
            data[file]["target%sIncrease%sDays" %(target, shiftDay)] = data[file]["target%sIncrease%sDays" %(target, shiftDay)].shift(-shiftDay)


        #lagged prices - also to be used in prediction
        for lag in lags:
            data[file]["close_lag_%s" %lag] = data[file][target].shift(lag)

        
        #technical indicators - A list is provided and indicators are only computed if and only if they are in the provided list

        #OVERLAP STUDIES

        #Bollinger Bands - BBANDS
        if "BBANDS" in indicators:
            for timeperiod in [3,5,10]:
                for ndevup in [2]:
                    for ndevdn in [2]:
                        for mattype in [0]:
                            upperband_name = "BBANDS_upper_Days_%s_Ndevup_%s_Ndevdn_%s_Mattype_%s" %(timeperiod, ndevup, ndevdn, mattype)
                            middleband_name = "BBANDS_middle_Days_%s_Ndevup_%s_Ndevdn_%s_Mattype_%s" %(timeperiod, ndevup, ndevdn, mattype)
                            lowerband_name = "BBANDS_lower_Days_%s_Ndevup_%s_Ndevdn_%s_Mattype_%s" %(timeperiod, ndevup, ndevdn, mattype)
                            data[file][upperband_name], data[file][middleband_name], data[file][lowerband_name] = talib.BBANDS(data[file][target], timeperiod, ndevup, ndevdn, mattype)
        
        #Hilbert Transform - Instantaneous Trendline - HT_TRENDLINE
        if "HT_TRENDLINE" in indicators:
            column_name = "HT_TRENDLINE"
            data[file][column_name] = talib.HT_TRENDLINE(data[file][target])

        #extra params - for moving averages
        moving_average_time_periods = [3,5,10,15,20,30,50]

        #MOVING AVERAGE - MA
        if "MA" in indicators:
            for timeperiod in moving_average_time_periods:
                for matype in [0]:
                    column_name = "MA_Days_%s_Mattype_%s" %(timeperiod, matype)
                    data[file][column_name] = talib.MA(data[file][target], timeperiod, matype)

        #Exponential Moving Average - EMA
        if "EMA" in indicators:
            for timeperiod in moving_average_time_periods:
                column_name = "EMA_Days_%s" %timeperiod
                data[file][column_name] = talib.EMA(data[file][target], timeperiod)

        #Double Exponential Moving Average - DEMA
        if "DEMA" in indicators:
            for timeperiod in moving_average_time_periods:
                column_name = "DEMA_Days_%s" %timeperiod
                data[file][column_name] = talib.DEMA(data[file][target], timeperiod)

        #Kaufman Adaptative moving average - KAMA
        if "KAMA" in indicators:
            for timeperiod in moving_average_time_periods:
                column_name = "KAMA_Days_%s" %timeperiod
                data[file][column_name] = talib.KAMA(data[file][target], timeperiod)
        
        #Simple Moving Average - SMA
        if "SMA" in indicators:        
            for timeperiod in moving_average_time_periods:
                column_name = "SMA_Days_%s" %timeperiod
                data[file][column_name] = talib.SMA(data[file][target], timeperiod)

        #T3 - Triple Exponential Moving Average - T3
        if "T3" in indicators:
            for timeperiod in moving_average_time_periods:
                for vfactor in [0]:
                    column_name = "T3_Days_%s_Vfactor_%s" %(timeperiod,vfactor)
                    data[file][column_name] = talib.T3(data[file][target], timeperiod, vfactor)

        # Triple Exponential Moving Average - TEMA
        if "TEMA" in indicators:
            for timeperiod in moving_average_time_periods:
                column_name = "TEMA_Days_%s" %timeperiod
                data[file][column_name] = talib.TEMA(data[file][target], timeperiod)

        # Triangular Moving Average
        if "TRIMA" in indicators:
            for timeperiod in moving_average_time_periods:
                column_name = "TRIMA_Days_%s" %timeperiod
                data[file][column_name] = talib.TRIMA(data[file][target], timeperiod)
        
        # Weighted Moving Average - WMA
        if "WMA" in indicators:
            for timeperiod in moving_average_time_periods:
                column_name = "WMA_Days_%s" %timeperiod
                data[file][column_name] = talib.WMA(data[file][target], timeperiod)

        # MOMENTUM INDICATORS

        #Aroon Oscillator - AROONOSC
        if "AROONOSC" in indicators:
            for timeperiod in [5,10,14,20,25]:
                column_name = "AROONOSC_Days_%s" %timeperiod
                data[file][column_name] = talib.AROONOSC(
                                    high = data[file]["high"],
                                    low = data[file]["low"],
                                    timeperiod = timeperiod)

        #Balance of Power - BOP
        if "BOP" in indicators:
            column_name = "BOP"
            data[file][column_name] = talib.BOP(data[file]["open"], data[file]["high"], data[file]["low"], data[file][target])

        # DX - Directional Movement Index
        if "DX" in indicators:
            for timeperiod in [5,10,14,20,25]:
                column_name = "DX_Days_%s" %timeperiod
                data[file][column_name] = talib.DX(data[file]["high"], data[file]["low"], data[file][target], timeperiod)
        


        #filter out dates again
        data[file] = data[file].loc[datetime.date(year=cutoffDate[2], month=cutoffDate[1], day=cutoffDate[0]):datetime.date(year=test_end[2], month=test_end[1], day=test_end[0])]

        missing = data[file].isna().sum()


        #In the end, drop rows with NaNs
        data[file] = data[file].dropna()


        #MAP TARGETS according to intervals provided
        for c in data[file].columns:
            if "target" in c:
                for interval in intervals:
                    data[file]["%s_%s" %(c,interval)] = pd.cut(data[file][c], bins=interval,
                    #rightmost value goes to the interval preceding (i.e., intervals are like ]x,y] )
                    #careful, always first value should be -math.inf when using right = True (this way no nans generated :)) 
                     right=True, labels=False)+1

                #delete original column after mapping
                data[file] = data[file].drop(columns=[c])

        #train / test split
        train[file] = data[file].loc[:datetime.date(year=train_end[2], month=train_end[1], day=train_end[0])]
        test[file] = data[file].loc[datetime.date(year=test_begin[2], month=test_begin[1], day=test_begin[0]):]


        print("Shape train:" , train[file].shape)
        print("start date train: ", train[file].iloc[0].name)
        print("end date train: ", train[file].iloc[-1].name)
        print("train date end: ", train_end)

        print("Shape test: ", test[file].shape)
        print("start date test: ", test[file].iloc[0].name)
        print("end date test: ", test[file].iloc[-1].name)

    return train, test 

###############
# X - Y SPLIT #
###############

"""
    Description of the function: split files into X (explanatory variables) and 
    Y (targets) for training and validation test.

    Inputs: 
        * data: dictionary file with preprocessed files
    
    Return:
        * X dictionary with Explanatory variables separated from data
        * Y dictionary with Targets variables separated from data
"""

def X_Y_split(data):

    print("\n\nNever forget to split your data into X and Y splits...")

    targets = []
    features = []

    X = {}
    Y = {}

    #extract targets and features
    for file in data:
        for c in data[file].columns:
            if "target" in c:
                targets.append(c)
            else:
                features.append(c)

        break

    #create X and Y dicts
    for file in data:
        print("Current stock been processed: ", file)
        X[file] = data[file][features]
        Y[file] = data[file][targets] 

    return X, Y