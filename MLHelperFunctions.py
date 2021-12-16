###############
# DESCRIPTION #
###############

"""
    In this scrpit, we provide helper functions such as plots, printing datasets, etc.
    to be used in the rest of the scripts.

"""

import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import itertools

############################
# GET DIGIST FROM A STRING #
############################

"""
    Description of the function: gets all the digits of a string and stores them in another string
    Input: string to be transformed
    Return: string with digits
"""

def get_digits(str1):
    c = ""
    for i in str1:
        if i.isdigit():
            c += i
    return c

##################
# DESCRIBE FILES #
##################

"""
    Description of the function: Provides a summary of the stock files stored in a dictionary (works for raw
    and clean data)

    Inputs: 
        * data: dictionary with pd.DataFrame files.
        * many: whether to print information of every stock or not
        * head: whether only the head should be displayed or not

"""

def describeFiles(data, many=False, head=False):

    for file in data:
        print("\n\n\nFile name: ", file)
        print("Index dtype: ", data[file].index.dtype)
        print("Columns: ", data[file].columns)
        print("Dtypes: ", data[file].dtypes)
        print("Shape: ", data[file].shape)
        #only print head if provided
        if head == True:
            print("\nHead: ", data[file].head(10))
        #only print all the files if provided
        if many == False:
            break

####################
# DESCRIBE TARGETS #
####################

"""
    Description of the function: prints all the targets in a file(s)
    and the (proportion of observations in each of the classes

    Inputs: 
        * data: dictionary with already preprocessed data
        * file: if desired information of which file specifically to print
        * normalize: whether to normalize the results from value counts or not
"""

def describeTargets(data, file=None, normalize=False):
    
    if file != None:
        print("\n\nStock: ", file[:-4])
        #check columns with the word "target" in their name
        for c in data[file].columns:
            if "target" in c:
                print("\nTarget Days: ", get_digits(c[:32]))
                print("Classes: ")
                classes = data[file][c].value_counts(sort=True, normalize=normalize)
                for cl in range(1,len(classes)+1):
                    inter = c[c.find("["):c.find("]")+1].strip('][').split(', ')
                    print("Interval [%s to %s]: %s " %(inter[cl-1],inter[cl], round(classes[cl],4)))
    else:
        for file in data:
            print("\n\n\nStock: ", file[:-4])
            for c in data[file].columns:
                if "target" in c:
                    print("\nTarget Days: ", get_digits(c[:32]))
                    print("Classes: ")
                    classes = data[file][c].value_counts(sort=True, normalize=normalize)
                    for cl in range(1,len(classes)+1):
                        inter = c[c.find("["):c.find("]")+1].strip('][').split(', ')
                        print("Interval [%s to %s]: %s " %(inter[cl-1],inter[cl], round(classes[cl],4)))


###########################################
# GENERATE (OR LOAD) EVALUATION DATAFRAME #
###########################################

"""
    Description of the function: Reads a results.csv path if it exits (where results will be appended).
    If results.csv does not exist yet (first experimentation), then create the file. 

    Inputs: 
     * path: path to results.csv file
     * file: results.csv
     * model_columns: variables that characterize a model (i.e., stock, model, preprocessing...)
     * evaluation_columns: evaluation metrics (i.e., accuracy, recall, f1...)
     * targets: target columns (i.e. Prediction of tomorrow's direction in interval [-inf, 0, inf])

    Output: evaluation dataframe (either newly created or loaded)
"""


def generate_evaluation_df(path, file, model_columns, evaluation_columns, targets):

    #if the path already exists, then simply return the results dataframe
    if file in os.listdir("results"):
        return pd.read_csv(path+"/"+file, sep=";", decimal=",")
    
    #if not, geneterate a dataframe of such a kind
    else:
        #list for the columns
        columns = []
        #append basic columns
        columns = columns + model_columns
        #loop over evaluation columns and train targets to combine them
        for target in targets:
            for evaluation in evaluation_columns:
                #make the name of the target cleaner
                m = re.search(r"\d", target).start()
                columns.append(evaluation+"_"+target[m:])

        #also append one column for training and prediction time & round
        columns += ["training_time", "prediction_time", "round"]

        #create dataframe
        evaluation = pd.DataFrame(columns=columns)

        #return dataframe
        return evaluation
