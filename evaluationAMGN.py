#IMPORTS
from numpy import true_divide
import sklearn
from joblib import dump, load
from tensorflow import keras
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
import plotly.express as px



#LOAD TEST DATA

#api key for alpha vantage
key = "Y7YNVWLSGI38NHOJ"

#data
symbols = ['AMGN']

#time series object
ts = TimeSeries(key, output_format="pandas")

saveData(symbols, ts) #Function located in Load_API_Data.py

basic_columns = ["open", "high", "low", "close", "adjustedClose", "volume", "dividendAmount", "SplitCoefficient"]
directory_load = "data"

raw_data = loadData(directory_load, basic_columns) #Function located in Load_API_Data.py


#load also train since it is needed for scaling
cutoffDate_train = [7,10,2009]
train_end = [31,12,2019]
cutoffDate = [1,1,2021]
test_end = [14,12,2021]

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
train, test = preprocess(raw_data, main_columns, cutoffDate_train,
 train_end, cutoffDate, test_end, shiftDays,
 indicators, intervals, lags, target)


#CALL for train-test splits
X_train, Y_train = X_Y_split(train) #Function located in PreprocessingML.py
X_test, Y_test = X_Y_split(test) #Function located in PreprocessingML.py


# EVALUATION PRINT
def print_evaluation(predictions, y_true):

    print("\n\nEvaluation Results: ")
    accuracy = accuracy_score(y_true, predictions)
    precision_macro = precision_score(y_true, predictions, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, predictions)

    print("- Accuracy score:\n ", accuracy)
    print("- Precision macro:\n", precision_macro)
    print("- Confusion Matrix:\n ", cm)


#LOAD BEST MODELS

# # Price direction 5 days - AMGN 
# # Ada Boost with the parameters shown in output:
model_path = "results/AMGN/models/4_5[-inf,0,inf]_AdaBoostClassifier__StandardScaler_PCA_SelectKBest_cv_default__model__learning_rate_1_1.1_1.5.joblib"
model = load(model_path)
column_name = "targetadjustedCloseIncrease5Days_[-inf, 0, inf]"
print("\n\n\nModel: ", model)
print("\nTarget: ", column_name)


predictions = model.predict(X_test["AMGN.csv"])
print_evaluation(Y_test["AMGN.csv"][column_name], predictions)


print(Y_test["AMGN.csv"][column_name].value_counts())
exit()


########################################################################################################################################################################################################################################

# # Price direction 10 days - AMGN --> >50% class precision for both classes in validation set
# #LSTM with following architecture:
# model_path = "results/AMGN/DL_Results/model/round_4_LSTM_10Days_[-inf, 0, inf]_accuracy_0.6"
# model = keras.models.load_model(model_path)
# column_name = "targetadjustedCloseIncrease10Days_[-inf, 0, inf]"
# print("\n\n\nModel: ")
# model.summary()
# print("\nTarget: ", column_name)

# #modify data
# #note that there will be less samples for evaluation of this one (due to the modifications - 60 time step)

# # Select columns for prediction (all but lags - redundancy)
# cols = []
# for c in X_test["AMGN.csv"].columns:
#     if "lag" not in c: #it was trained without lags, so...
#         cols.append(c)

# x_train = X_train["AMGN.csv"][cols]
# x_test = X_test["AMGN.csv"][cols]


# original_x_t_train = X_train["AMGN.csv"][cols].copy()
# original_x_test = X_test["AMGN.csv"][cols].copy()
# original_y_train = Y_train["AMGN.csv"].copy()
# original_y_test = Y_test["AMGN.csv"].copy()

# #scale data
# sc = StandardScaler()
# training_set_scaled = sc.fit_transform(x_train) 
# test_set_scaled = sc.transform(x_test)

# #reshape
# n = 60
# n_rows_test = test_set_scaled.shape[0]
# n_columns_test = test_set_scaled.shape[1]


# X_test = []

# #create lagged test set
# for i in range(n+1,n_rows_test+1):
#     X_test.append(test_set_scaled[i-n:i,:])

# #convert to np array and reshape 
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_columns_test))

# #predict
# predicted_stock_price = model.predict(X_test)
# predicted_stock_price_labels = np.argmax(predicted_stock_price, axis=-1)


# target_test_labels = original_y_test.iloc[n:,-1] -1

# print_evaluation(target_test_labels, predicted_stock_price_labels)


########################################################################################################################################################################################################################################



# Market opportunity 10 days - AMGN 
#LSTM with following architecture:
model_path = "results/AMGN/DL_Results/model/round_5_LSTM_withLags_10Days_[-inf, -1.5, 1.5, inf]_accuracy_0.5"
model = keras.models.load_model(model_path)
column_name = "targetadjustedCloseIncrease10Days_[-inf, -1.5, 1.5, inf]"
print("\n\n\nModel: ")
model.summary()
print("\nTarget: ", column_name)

#modify data
#note that there will be less samples for evaluation of this one (due to the modifications - 60 time step)

# Select columns for prediction (all but lags - redundancy)
cols = []
for c in X_test["AMGN.csv"].columns:
    # if "lag" not in c: #it was trained with lags, so this should be commented
        cols.append(c)

x_train = X_train["AMGN.csv"][cols]
x_test = X_test["AMGN.csv"][cols]


original_x_t_train = X_train["AMGN.csv"][cols].copy()
original_x_test = X_test["AMGN.csv"][cols].copy()
original_y_train = Y_train["AMGN.csv"].copy()
original_y_test = Y_test["AMGN.csv"].copy()

#scale data
sc = StandardScaler()
training_set_scaled = sc.fit_transform(x_train) 
test_set_scaled = sc.transform(x_test)

#reshape
n = 60
n_rows_test = test_set_scaled.shape[0]
n_columns_test = test_set_scaled.shape[1]


X_test = []

#create lagged test set
for i in range(n+1,n_rows_test+1):
    X_test.append(test_set_scaled[i-n:i,:])

#convert to np array and reshape 
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_columns_test))

#predict
predicted_stock_price = model.predict(X_test)
predicted_stock_price_labels = np.argmax(predicted_stock_price, axis=-1)


target_test_labels = original_y_test.iloc[n:,-2] -1

print_evaluation(target_test_labels, predicted_stock_price_labels)

