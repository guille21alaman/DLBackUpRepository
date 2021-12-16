#add dummy variables to results_clean.csv file

import pandas as pd
import numpy as np
from Load_API_Data import *
from MLHelperFunctions import *
from MLPreprocessing import *
from MLModelling import * 
import math
from cleanResults import *
import random


#clean results first
clean_results()

#read results_clean file
results = pd.read_csv("results/results_clean.csv", sep=";", decimal=",")

#read data and preprocess as done for training
#region

basic_columns = ["open", "high", "low", "close", "adjustedClose", "volume", "dividendAmount", "SplitCoefficient"]
directory_load = "data"
raw_data = loadData(directory_load, basic_columns)

cutoffDate = [1,1,2010]
train_end = [31,12,2019]
test_begin = [1,1,2020]
test_end = [31,12,2020]

target = "adjustedClose"
main_columns = ["open", "high", "low", "close", "adjustedClose", "volume", "dividendAmount", "SplitCoefficient"] 

shiftDays = [1,5,10]

lags = [1,2,3,4,5,6,7,8,9,10,20,50]

indicators = ["BBANDS", "HT_TRENDLINE", "MA",
 "SMA", "DEMA", "KAMA", "SMA", "T3", "TEMA", "TRIMA", "WMA",
 "DX", "AROOSC", "BOP", "EMA"]

intervals = [
    [-math.inf, -1.5, 1.5, math.inf], # catch increases that are worthy
    [-math.inf, 0, math.inf] #up or down   
]

train, test = preprocess(raw_data, main_columns, cutoffDate,
 train_end, test_begin, test_end, shiftDays,
 indicators, intervals, lags, target)

X_train, Y_train = X_Y_split(train) #Function located in PreprocessingML.py
X_test, Y_test = X_Y_split(test) #Function located in PreprocessingML.py

 #endregion




# three new rows per stock are going to be added to the results dataframe
## 1) Dummy predictions: predict the majority class on the training set
## This idea represents a really dummy strategy in which the investor basically expects
## the most common event to be repeated in future transactions
## 2) Random predictions: this represents an strategy in which the investor basically
## assigns one of the 3 classes from the training set randomly. 
## In the case of price increase/decrease, this will be equivalent
## to an investor randomly predicting the price to go up or down


#loop over stocks
for stock in Y_test: #Y_test could be substituted by X_test, X_train or Y_train
    row_dummy = {}
    row_random = {}

    row_dummy["Stock"], row_dummy["Model"], row_dummy["round"] = stock, "Dummy", 0
    row_random["Stock"], row_random["Model"], row_dummy["round"] = stock, "Random", 0

    #loop over targets
    for target in Y_test[stock].columns:

        column_name = target.replace("targetadjustedCloseIncrease", "")

        #needed later
        list_evaluation_columns = []
        for c in results.columns:
            if column_name in c:
                if "Accuracy" in c: 
                    list_evaluation_columns.append(c)
                elif ("Precision" in c) or ("Preicision" in c):
                    list_evaluation_columns.append(c)
                else:
                    continue
        
        #dummy predictions based on the most common class of train set (size of test set)
        dummy_predictions = np.repeat(Y_train[stock][target].mode()[0], Y_test[stock][target].shape[0])
        
        random.seed(7)

        #random predictions based on classes of train set
        random_predictions = []
        for x in range(Y_test[stock][target].shape[0]):
            random_predictions.append(np.random.choice(Y_train[stock][target].unique()))

        #compute necessary variables
        accuracy_dummy = accuracy_score(Y_test[stock][target], dummy_predictions)
        precision_micro_dummy = precision_score(Y_test[stock][target], dummy_predictions, average="micro")
        precision_macro_dummy = precision_score(Y_test[stock][target], dummy_predictions, average="macro")
        precision_weighted_dummy = precision_score(Y_test[stock][target], dummy_predictions, average="weighted")

        #append variables into evaluation list for storing in results.csv
        evaluation_list_dummy = []
        evaluation_list_dummy.append(accuracy_dummy)
        evaluation_list_dummy.append(precision_micro_dummy)
        evaluation_list_dummy.append(precision_macro_dummy)
        evaluation_list_dummy.append(precision_weighted_dummy)
        
        #now add to the dictionary where data is stored
        count = 0
        for c in list_evaluation_columns:
            row_dummy[c] = evaluation_list_dummy[count]
            count += 1

        #compute necessary variables
        accuracy_random = accuracy_score(Y_test[stock][target], random_predictions)
        precision_micro_random = precision_score(Y_test[stock][target], random_predictions, average="micro")
        precision_macro_random = precision_score(Y_test[stock][target], random_predictions, average="macro")
        precision_weighted_random = precision_score(Y_test[stock][target], random_predictions, average="weighted")

        #append variables into evaluation list for storing in results.csv
        evaluation_list_random = []
        evaluation_list_random.append(accuracy_random)
        evaluation_list_random.append(precision_micro_random)
        evaluation_list_random.append(precision_macro_random)
        evaluation_list_random.append(precision_weighted_random)
        
        #now add to the dictionary where data is stored
        count = 0
        for c in list_evaluation_columns:
            row_random[c] = evaluation_list_random[count]
            count += 1


    results = results.append(row_dummy, ignore_index=True)
    results = results.append(row_random, ignore_index=True)


results.to_csv("results/results_clean.csv", sep=";", decimal = ",", index=None) 
results.to_excel("results/results_clean.xlsx", index=False)