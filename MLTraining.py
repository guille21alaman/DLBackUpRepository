#TEST
#flag for testing during deployment
testing = False

###########
# IMPORTS #
###########
#region

import warnings
warnings.filterwarnings('ignore')


from sklearn import tree
from Load_API_Data import *
from MLHelperFunctions import *
from MLPreprocessing import *
from MLModelling import * 


import multiprocessing 
import math
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Lasso
from random import randint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import itertools
import datetime
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from joblib import dump, load
import random
from scipy.stats import uniform
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from alpha_vantage.timeseries import TimeSeries
from sklearn.svm import SVC


#endregion

# --------------------------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------------------------------------#

#####################
# CALL WHOLE METHOD #
#####################

print("#############################################\nStarting ML Experimentation Script...\n#############################################\n")

#QUERY DATA API AND LOAD
#region

# PARAMS QUERY FUNCTION
#api key for alpha vantage
key = "Y7YNVWLSGI38NHOJ"

#data
symbols = ['AAPL', 'TSLA', 'AMGN']

#time series object
ts = TimeSeries(key, output_format="pandas")

#CALL function to save data (only need to do this once :) ) 
print("#############################################\nQuerying data from API\n#############################################\n")
# saveData(symbols, ts) #Function located in Load_API_Data.py

# PARAMS LOAD FUNCTION
basic_columns = ["open", "high", "low", "close", "adjustedClose", "volume", "dividendAmount", "SplitCoefficient"]
directory_load = "data"

#LOAD QUERIED DATA
#CALL loadDataFunction
print("#############################################\nLet's load some of the data we just queried...\n#############################################")
raw_data = loadData(directory_load, basic_columns) #Function located in Load_API_Data.py

#CALL function for describing files
# describeFiles(raw_data) #Function located in HelperFunctions.py

#endregion

# --------------------------------------------------------------------------------------------------------------#

#PREPROCESSING
print("\n#############################################\nStarting preprocessing...\n#############################################\n")

# region

# PARAMS preprocessing
#dates for the split
cutoffDate = [1,1,2010]
train_end = [31,12,2019]
test_begin = [1,1,2020]
test_end = [31,12,2020]

#features to be used
target = "adjustedClose"
main_columns = ["open", "high", "low", "close", "adjustedClose", "volume", "dividendAmount", "SplitCoefficient"] 

#shift days for the target increase
shiftDays = [1,5,10]

#for the prices to use as predictors
lags = [1,2,3,4,5,6,7,8,9,10,20,50]

#technical indicators
indicators = ["BBANDS", "HT_TRENDLINE", "MA",
 "SMA", "DEMA", "KAMA", "SMA", "T3", "TEMA", "TRIMA", "WMA",
 "DX", "AROOSC", "BOP", "EMA"]

#intervals for the targets
intervals = [
    [-math.inf, -1.5, 1.5, math.inf], # catch increases that are worthy
    [-math.inf, 0, math.inf] #up or down   
]

#CALL preprocessing function - # Function located in PreprocessingML.py
train, test = preprocess(raw_data, main_columns, cutoffDate,
 train_end, test_begin, test_end, shiftDays,
 indicators, intervals, lags, target)


#CALL for train-test splits
X_train, Y_train = X_Y_split(train) #Function located in PreprocessingML.py
X_test, Y_test = X_Y_split(test) #Function located in PreprocessingML.py


#CALL to analyze class imbalance
print("\n#############################################\nShow Targets:\n#############################################\n")
describeTargets(train, file=None, normalize=True) # Function located in HelperFunctions.py
describeTargets(test, file="AAPL.csv", normalize=True) # Function located in HelperFunctions.py
# print("\n#############################################\nShow X_train:\n#############################################\n")
# describeFiles(X_train)de

#endregion

# --------------------------------------------------------------------------------------------------------------#


#LOAD EVALUATION
print("\n#############################################\nLoading/creating evaluation file...\n#############################################\n")

#region
#PARAMS
path_results = "results"
file_results = "results.csv"

# Model columns - Information about each model to be appended to the main evaluation dataframe
model_columns = ["Stock", "Model", "Preprocessing", "CrossValidation", "Parameters"]
# Evaluation models - Evaluation measures for each target
evaluation_columns = [
    "Accuracy",
    "RecallMicro", "RecallMarco", "RecallWeighted",
    "PreicisionMicro", "PrecisionMarco", "PrecisionWeighted", 
    "F1Micro", "F1Macro", "F1Weighted"]


#CALL dataframe generator/loader for evaluation
#extract target columns into a list
for symbol in Y_train:
    targets = Y_train[symbol].columns.to_list()
    break #because we only need to extract targets once

#then actually call the function
results = generate_evaluation_df(path_results, file_results, model_columns, evaluation_columns, targets) # Function located in HelperFunctions.py
#endregion


#TRAINING 
print("\n#############################################\nTraining starts...\n#############################################\n")

#region
#PARAMS

#experimentation round: see experimentsDesign.md for more on each round
exp_round = 9

#whether to save confusion matrices or not (accruacy threshold) and cv results
#the threshold is calculated as follows:
## threshold accuracy = dummy_accuracy + threshold 
## dummy accuracy = accuracy of the model that predicts always training majority class
if testing == True:
    threshold = 0.0
else:
    threshold = 0.1 #10% better than dummy accuracy

#parameter to save every model above (i.e., all mdoels with accuracy over 90% will be printed (regardless of the dummy accuracy))
emergency_threshold = 0.9 


#MULTITRAINING
cvs = [
    TimeSeriesSplit(n_splits=5)
]

preprocessings = [
    # [StandardScaler(), PCA(), SelectKBest()],
    [StandardScaler(), SelectKBest()]
]

classifiers = [
    MLPClassifier(),
    # AdaBoostClassifier()
]


#make "manual" randomized search by generating random lists of parameters :) 
params = {

    #TimeSeries split 5 - Preprocessing 1 - MLP
    # 0: {
    #     "%s__n_components"  %(type(preprocessings[0][1]).__name__): [20,65,70,None],
    #     "%s__k" %(type(preprocessings[0][2]).__name__): [80,"all"],
    #     "model__max_iter": [700,100,1500],
    #     "model__batch_size": [150,175,200],
    #     "model__solver": ["adam"],
    #     "model__activation": ["tanh"]
    # },


    #     #TimeSeries split 5 - Preprocessing 1 - AdaBoost
    # 0: {
    #     # "%s__n_components"  %(type(preprocessings[0][1]).__name__): [20,65,70,None],
    #     "%s__k" %(type(preprocessings[0][1]).__name__): [50,"all"],
    #     "model__n_estimators": [50,100,200],
    #     # "model__learning_rate": [1,1.1,1.5]
    # },

    #     #TimeSeries split 5- Preprocessing 2 - MLP
    0: {
        "%s__k" %(type(preprocessings[0][1]).__name__): [50,"all"],
        "model__max_iter": [500],
        "model__batch_size": [64,200],
        "model__solver": ["adam"],
        "model__activation": ["relu"]
    },

    # # #     #TimeSeries split 5- Preprocessing 2  - AdaBoost
    # 3: {
    #     "%s__k" %(type(preprocessings[0][2]).__name__): [80, "all"],
    #     "model__n_estimators": [350,400,500],
    #     "model__learning_rate": [1,1.1,1.5]
    # },

    # Add more combinations here
}


# CALL TRAINING - #Functions in ModellingML.py
# loop over stocks and apply same multitraining to all of them
print("\n Let the fun begin :)...")
for symbol in train:

    print("\n\n\n\n\n\n\n")
    print("-----------------------------------------------------")

    #CALL MULTIPLE_TRAINING FUNCTION - This function and its dependent functions called in ModellingML.py
    results = multiple_training(
    cvs, preprocessings, classifiers,
    params, X_train[symbol], Y_train[symbol],
    X_test[symbol], Y_test[symbol], symbol, results, model_columns, targets, evaluation_columns,
    threshold, emergency_threshold, intervals, shiftDays, symbols, exp_round)

    print("\n\n\n\n\n\n\n")
    print("-----------------------------------------------------")
    if testing == True:
        print(results)
        print("Well done!")
        exit()


#save results
results.to_csv("results/results.csv", sep=";", decimal=",", index=False)

print("Show final results:\n ", results)
#endregion


#endregion
# --------------------------------------------------------------------------------------------------------------#

##################
# CLOSING SCRIPT #
##################
#region
print("-----------------------")
print("\n\n\n\n\n\n\n")
print("Well done!")

#Make a beep when script is done ()
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
#endregion

# --------------------------------------------------------------------------------------------------------------#
