#TEST
#flag for testing during deployment
testing = True

###########
# IMPORTS #
###########
#region

from re import S
import warnings

warnings.filterwarnings('ignore')


from sklearn import tree
from Load_API_Data import *
from MLHelperFunctions import *
from MLPreprocessing import *

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

import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras import Input, Model, optimizers, callbacks
from tensorflow.keras import layers

from tensorflow.keras import utils


import numpy as np
import pandas as pd
import os, datetime
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
print('Tensorflow version: {}'.format(tf.__version__))

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import warnings
warnings.filterwarnings('ignore')

def plot_confusion_matrix(cm, classes, path, display=False, normalize=False, title='Confusion matrix', cmap=plt.cm.Reds):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")

    # print(cm)
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #colormap between 0 and 1 for normalized
    if normalize == True:
        plt.clim(0,1)
    #else leave it automatically
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path)

    #show only if display true
    if display == True:
      plt.show()

    else:
      plt.close()

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

#To have the same number of days as in ML experimentation, we need to take some more days (as many as time steps will be used in the LSTM net) -
#The others will be deleted when generating the tensors
n = 60 #this is a bit tricky to do --> check that is alright for any change!

#Note that in this case, some entries from train test will be used as time-steps for test phase which is not optimal. However, for the shake of keeping the
#comparability with respect to traditional ML models, I decided to keep them

# PARAMS preprocessing
#dates for the split
cutoffDate = [7,10,2009]
train_end = [31,12,2019]
test_begin = [4,10,2019]
test_end = [31,12,2020]

#features to be used
target = "adjustedClose"
main_columns = ["open", "high", "low", "close", "adjustedClose",
 "volume", "dividendAmount", "SplitCoefficient"] 

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
Xtrain, Ytrain = X_Y_split(train) #Function located in PreprocessingML.py
Xtest, Ytest = X_Y_split(test) #Function located in PreprocessingML.py



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
for symbol in Ytrain:
    targets = Ytrain[symbol].columns.to_list()
    break #because we only need to extract targets once

#then actually call the function
results = generate_evaluation_df(path_results, file_results, model_columns, evaluation_columns, targets) # Function located in HelperFunctions.py
#endregion

# until here, same as before


#to store training/test accuracy
results_train_test = pd.DataFrame(columns=["model", "stock", "round", "train_accuracy", "test_accuracy"])

#row to append to ML results
row = {} 
row_train_test = {}

 
# Parameters Model
epochs = 1
batch_size = 64
loss = "categorical_crossentropy"
optimizer = "adam"
experiment_round = 7
model_name = "Transformer"
dropout = 0.2
mlp_dropout = 0.2


# Select columns for prediction (all but lags - redundancy)
cols = []
for c in Xtrain["AAPL.csv"].columns:
    # if "lag" not in c:
    cols.append(c)



#loop over all the stocks 
for stock in symbols: 

    stock = stock + ".csv"

    #naming for results
    row["Stock"] = stock
    row["Model"] = model_name
    row["round"] = experiment_round

    #loop over targets
    for target in range(Ytrain[stock].shape[1]):


        row_train_test["model"] = model_name
        row_train_test["round"] = stock
        row_train_test["stock"] = experiment_round

        #naming of the column like in results file
        m = re.search(r"\d", Ytest[stock].columns[target]).start()
        column_name = Ytest[stock].columns[target][m:]

        list_evaluation_columns = []
        for c in results.columns:
            if column_name in c:
                if "Accuracy" in c: 
                    list_evaluation_columns.append(c)
                elif ("Precision" in c) or ("Preicision" in c):
                    list_evaluation_columns.append(c)
                else:
                    continue

        #create a list of ordered labels according to the number of classes
        labels = list(range(1,len(Ytrain[stock].iloc[:,target].unique())+1))

        #get name of the interval
        column_name = Ytest[stock].iloc[:,target].name
        #find where interval starts
        index_first = column_name.find("[")
        #convert to an actual list
        labels_str = column_name[index_first:].strip('][').split(', ')

        #create index and columns for the nice confusion matrix df
        index, column, plot_labels = [], [], []
        for lab in range(len(labels_str)-1):
            index.append("true: [%s%%, %s%%]" %(labels_str[lab], labels_str[lab+1]))
            column.append("pred: [%s%%, %s%%]" %(labels_str[lab], labels_str[lab+1]))
            plot_labels.append("[%s%%, %s%%]" %(labels_str[lab], labels_str[lab+1]))


        #restart all again
        x_train = Xtrain[stock][cols]
        x_test = Xtest[stock][cols]

        #copies to be used later for evaluation
        original_x_t_train = Xtrain[stock][cols].copy()
        original_x_test = Xtest[stock][cols].copy()
        original_y_train = Ytrain[stock].copy()
        original_y_test = Ytest[stock].copy()

        print("\n\n\n\n\n\nNew target: ", Ytrain[stock].columns[target], "\n\n")

        # print("Current symbol: ", stock)

        # print("The original datasets look as follows: ")

        # print("X_train: ", original_x_t_train.iloc[n:,], "\nwith shape: ", original_x_t_train.iloc[n:,].shape)
        # print("y_train: ", original_y_train.iloc[n:,], "\n with shape: ", original_y_train.iloc[n:,].shape)
        
        
        #Feature scaling
        #scaling features: fit and transform in train, only transform on test set
        sc = StandardScaler()
        training_set_scaled = sc.fit_transform(x_train) 
        test_set_scaled = sc.transform(x_test)

        # print("\nAfter scaling, the x set looks as follows: ")
        # print("X_train: ", training_set_scaled[n:,], "with shape: ", training_set_scaled[n:,].shape)

        #Expected shape of LSTM: 3D tensor; (batch_size, time steps and features)
        #Reshaping data structure with n time steps and 1 output
        #define n time steps. Explanation https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
        n = 60
        n_rows_train = training_set_scaled.shape[0]
        n_rows_test = test_set_scaled.shape[0]
        n_columns_train = training_set_scaled.shape[1]
        n_columns_test = test_set_scaled.shape[1]


        #empty lists for X train and Y train 
        X_train = []
        y_train = []

        #create lagged targets and time-step windows of size n 
        for i in range(n+1,n_rows_train+1): #+1 to be consistent with the target encoding
            X_train.append(training_set_scaled[i-n:i,:]) 

        #empty list for X test and Y test
        X_test = []
        y_test = []

        #create lagged test set
        for i in range(n+1,n_rows_test+1):
            X_test.append(test_set_scaled[i-n:i,:])
        
        #convert to np array and reshape
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_columns_train))
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_columns_test))


        # print("\nAfter reshaping:")
        # print("X_train: ", X_train, "\nwith shape: ", X_train.shape)
  
        #parameter for dense layer
        units_last_dense = len(np.unique(original_y_train.iloc[:,target]))

        #input shape for the model: (batch size, time steps, features) - perfectly compatible with RNN inputs (as in LSTM)
        #in this case, we extract the (time steps, features) combination. The model already expect data to be fed by batches
        input_shape = X_train.shape[1:]
    


        # Model Architecture
        # This is the Transformer architecture from Attention Is All You Need, applied to timeseries instead of natural language.
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Normalization and Attention
            x = layers.LayerNormalization(epsilon=1e-6)(inputs)
            x = layers.MultiHeadAttention(
                
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(x, x)
            x = layers.Dropout(dropout)(x)
            res = x + inputs

            # Feed Forward Part
            x = layers.LayerNormalization(epsilon=1e-6)(res)
            x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
            x = layers.Dropout(dropout)(x)
            x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
            return x + res

        # buld model 
        def build_model(
            input_shape,
            head_size,
            num_heads,
            ff_dim,
            num_transformer_blocks,
            mlp_units,
            dropout=dropout,
            mlp_dropout=mlp_dropout,
            n_classes=units_last_dense
         ):
            inputs = Input(shape=input_shape)
            x = inputs
            for _ in range(num_transformer_blocks):
                x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

            x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
            for dim in mlp_units:
                x = layers.Dense(dim, activation="relu")(x)
                x = layers.Dropout(mlp_dropout)(x)
            outputs = layers.Dense(n_classes, activation="softmax")(x)
            return Model(inputs, outputs)

        #BUILD MODEL
        model = build_model(
                input_shape,
                head_size=150,
                num_heads=8,
                ff_dim=8,
                num_transformer_blocks=3,
                mlp_units=[512],
                mlp_dropout=0.4,
                dropout=0.2,
        )

        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizers.Adam(learning_rate=1e-4),
            metrics=["accuracy"],
        )

        model.summary()

        callbacks_ = [callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

        print("#########################################\nTraining starts...\n#############################################") 
        target_train = utils.to_categorical(original_y_train.iloc[n:,target] -1) #-1 to be consistent with keras class organization 0,1,2...,n
        target_test = utils.to_categorical(original_y_test.iloc[n:,target]- 1)

        # print("Eventually, the target looks as follows: ")
        # print("y_train: ", target_train, "\nwith shape: ", target_train.shape)

        #to avoid class imbalance - give inverse weights
        class_weights_ = dict(1 - (Ytrain[stock].iloc[:,target].value_counts()/len(Ytrain[stock])))
        class_weights = {}

        for k in class_weights_:
            class_weights[k-1] = class_weights_[k]

        # Fitting the RNN to the Training set
        print("\n\n")
        print("Shape x: ", X_train.shape)
        print("Shape y: ", target_train.shape)
        print("\n")
        model.fit(X_train, target_train, epochs = epochs, batch_size = batch_size,
            validation_split=0.2, callbacks=callbacks_, class_weight=class_weights) #note that under epochs you can see how many batches were processed so far
        print("\n\n")

        print("#########################################\nEvaluation starts...\n#############################################")

        print("\nEvaluation summary: ")

        #prediction on X_train
        predicted_stock_price = model.predict(X_train)
        predicted_stock_price_labels = np.argmax(predicted_stock_price, axis=-1)
        target_train_labels = original_y_train.iloc[n:,target] -1
        training_accuracy = accuracy_score(target_train_labels, predicted_stock_price_labels)
        print("- Training accuracy: " ,training_accuracy, "\n")

        print("Training confusion matrix :\n",
         confusion_matrix(target_train_labels, predicted_stock_price_labels))
        print("Most frequent class training: ", target_train_labels.mode()[0])


        #prediction on X_test and evaluation
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price_labels = np.argmax(predicted_stock_price, axis=-1)
        target_test_labels = original_y_test.iloc[n:,target] -1


        #accuracy evaluation
        test_accuracy = accuracy_score(target_test_labels, predicted_stock_price_labels)
        print("- Test accuracy :" , test_accuracy)

        #precision evaluation
        if "1.5" in  Ytrain[stock].columns[target]:
            test_precision_micro = recall_score(target_test_labels, predicted_stock_price_labels,
             average="micro", zero_division=0)
            test_precision_macro = recall_score(target_test_labels, predicted_stock_price_labels,
             average="macro", zero_division=0)
            test_precision_weighted = recall_score(target_test_labels, predicted_stock_price_labels,
             average="weighted", zero_division=0)

            #print precisions
            print("- Test precision micro: ", test_precision_micro)
            print("- Test precision macro: ", test_precision_macro)
            print("- Test precision weighted: ", test_precision_weighted)

        
        #append variables into evaluation list for storing in results.csv
        evaluation_list = []
        evaluation_list.append(test_accuracy)
        evaluation_list.append(test_precision_micro)
        evaluation_list.append(test_precision_macro)
        evaluation_list.append(test_precision_weighted)
        
        #now add to the dictionary where data is stored
        count = 0
        for c in list_evaluation_columns:
            row[c] = evaluation_list[count]
            count += 1

        #confusion matrix
        cm = confusion_matrix(target_test_labels, predicted_stock_price_labels)
        print("- Confusion matrix: \n", cm)

        #naming for confusion matrix & model and architecture
        cm_name = "round_" + "%s" %experiment_round + "_" + model_name + "_" + column_name.replace("targetadjustedCloseIncrease", "") + "_accuracy_" +  "%s" %round(test_accuracy,1)
        
        #save model 
        model.save("results/%s/DL_Results/model/%s" %(stock.replace(".csv", ""), cm_name))    

        #save confusion matrices
        #not normalized
        plot_confusion_matrix(
            cm,
            plot_labels,
            "results/%s/DL_Results/%s.jpg" %(stock.replace(".csv", ""), cm_name),
            display=False,
            normalize=False,
            title = "Round\n" + "%s" %experiment_round + "\n" + model_name + "\n" + column_name.replace("targetadjustedCloseIncrease", "") + "Accuracy:\n" +  "%s" %round(test_accuracy,1)
        )

        #normalized
        plot_confusion_matrix(
            cm,
            plot_labels,
            "results/%s/DL_Results/%s_normalized.jpg" %(stock.replace(".csv", ""), cm_name),
            display=False,
            normalize=True,
            title = "Round\n" + "%s" %experiment_round + "\n" + model_name + "\n" + column_name.replace("targetadjustedCloseIncrease", "") + "Accuracy:\n" +  "%s" %round(test_accuracy,1)        )

        print("Naming for architecture screenshot: ", cm_name)
        
        row_train_test["train_accuracy"] = training_accuracy
        row_train_test["test_accuracy"] = test_accuracy

        results_train_test = results_train_test.append(row_train_test, ignore_index=True)
        results_train_test.to_csv("results/results_train_test_round_%s.csv" %experiment_round, sep=";", decimal=",", index=None)
        
    results = results.append(row, ignore_index=True)

    #export results
    results.to_csv("results/results.csv", sep=";", decimal = ",", index=None)
