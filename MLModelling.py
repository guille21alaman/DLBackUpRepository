###############
# DESCRIPTION #
###############

"""
    In this scrpit, all functions needed for training are described in detail.
"""

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

import multiprocessing 
import pandas as pd
import numpy as np
import re
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from random import randint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import itertools
from MLHelperFunctions import get_digits

#########################
# PLOT CONFUSION MATRIX #
#########################

"""
    TODO 
"""

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


#####################
# GRID SEARCH MODEL #
#####################

"""
    Description of the function: Core function for training. Grid search training for a model combination.
    Results are dynamically stored in the variable results (which is the core for results.csv)

    Inputs: 
    * cv: cross validation object (i.e., TimeSeriesSplit(n_splits=5))
    * preprocessing: list of preprocessing objects to be run in a pipeline (i.e. [StandardScaler(), PCA()...])
    * model: model to be run in the grid search (i.e. DecisionTreeClassifier())
    * model_params: Parameters for the model (i.e. {criterion: ["giny", "entropy"]})
    * X_train, Y_train, X_test, Y_test: corresponding dataframes to be used for training and validation 
    * symbol: current stock being processed
    * verbose: amount of information to be print during the grid search training (the higher, the more information)
    * results: variable storing temporal results of all models trained, 
    * model_columns: list of elements that define a model (i.e, stock, cv, preprocessing, etc. )
    * targets: list of targets to be predicted
    * evaluation_columns: list of evaluation metrics
    * normalize_cm: whether to normalize the confusion matrix built during evaluation or not
    * threshold: threshold to store results (i.e. confusion matrix and detailed evaluation) for a model
    (example: 25% This quantity is added to the dummy threshold and forms the accuracy threshold)
    * emergency_threshold: it could be that the accuracy threshold is > 100%, therefore we define an emergency threshold 
    (i.e. 90% accuracy. If validation accuracy is >= 90%, then the detailed mdoel results are stored)

"""

def grid_search_model(cv, preprocessing, model, model_params, X_train, Y_train, X_test, Y_test, symbol, 
    results, model_columns, targets, evaluation_columns, threshold, emergency_threshold, exp_round, verbose=2):

    print("\n\n\nStock %s" %symbol)
    print("Model: %s" %model.__class__.__name__)
    print("Preprocessing: %s" %preprocessing)
    print("CrossValidation %s" %cv)
    print("Parameters %s \n" %model_params)

    #evaluation list (to be used later in a loop)
    evaluation_list = []

    #start a list to store pipeline steps
    pipeline_steps = []

    #create list of tuples with preprocessing steps for grid search pipeline
    for p in preprocessing:
        tuple = (type(p).__name__, p)
        pipeline_steps.append(tuple)
    
    #add also the model to the pipeline
    pipeline_steps.append(("model", model))

    #build the pipeline
    pipeline = Pipeline(steps=pipeline_steps)

    identifier= 0

    #loop over the targets
    #loop over length of y test columns (i.e., number of targets)
    for target in range(Y_test.shape[1]):

        #search object
        search = GridSearchCV(estimator=pipeline, param_grid=model_params, cv=cv, n_jobs=-1, verbose=verbose)

        #fit
        train_start = datetime.datetime.now()
        search.fit(X_train, Y_train.iloc[:,target])
        train_total = datetime.datetime.now() - train_start

        #predict and return accuracy
        pred_start = datetime.datetime.now()
        predictions = search.predict(X_test)
        pred_total = datetime.datetime.now() - pred_start

        #evaluation
        print("Target: ", Y_test.iloc[:,target].name)

        #extract more beautiful column name
        m = re.search(r"\d", Y_test.columns[target]).start()
        column_name = Y_test.columns[target][m:]

        #measures to compute for every model: accuracy, recall, precision, f1score
        #accuracy 
        accuracy = accuracy_score(Y_test.iloc[:,target], predictions)
        evaluation_list.append(accuracy)

        #recall
        # zero_divison --> same idea as for precision
        evaluation_list.append(recall_score(Y_test.iloc[:,target], predictions, average="micro", zero_division=0))
        evaluation_list.append(recall_score(Y_test.iloc[:,target], predictions, average="macro", zero_division=0))
        evaluation_list.append(recall_score(Y_test.iloc[:,target], predictions, average="weighted", zero_division=0))

        #precision
        #zero_divison = 0 --> Meaning: those classes that have True_positive = 0 and False_positive = 0
        # have a 0-division when computing precision: tp / tp + fp. Thus, we decide to return a 0 for those values (since we DO care about them)
        evaluation_list.append(precision_score(Y_test.iloc[:,target], predictions, average="micro", zero_division=0))
        evaluation_list.append(precision_score(Y_test.iloc[:,target], predictions, average="macro", zero_division=0))
        evaluation_list.append(precision_score(Y_test.iloc[:,target], predictions, average="weighted", zero_division=0))

        #f1
        #same as for precision and recall
        evaluation_list.append(f1_score(Y_test.iloc[:,target], predictions, average="micro", zero_division=0))
        evaluation_list.append(f1_score(Y_test.iloc[:,target], predictions, average="macro", zero_division=0))
        evaluation_list.append(f1_score(Y_test.iloc[:,target], predictions, average="weighted", zero_division=0))

        #measures to compute only if above threshold: confusion matrix and cross-validation results
        
        #define threshold: 
        #first define a "dummy classifier" that predicts always majority class from train set, and calculate its accuracy
        #frequency of majority (train) class / length of the test set
        #compute most frequent class train
        most_frequent_train = Y_train.iloc[:,target].mode()[0]
        #array with dummy predictions
        dummy_predictions = np.full(shape = len(Y_test.iloc[:,target]), fill_value = most_frequent_train)
        #accuracy dummy
        accuracy_dummy = accuracy_score(Y_test.iloc[:,target], dummy_predictions)

        #threshold confusion matrix and cv results 
        threshold_cf = accuracy_dummy + threshold


        if len(model_params) == 0:
            model_params_str = ""
        else: 
            for object in model_params:
                s = model_params[object]
                model_params_str =  "_" + object + "_" + ' '.join([str(elem) for elem in s]).replace(" ", "_")
                

            
        #if threshold over 90% accuracy, reduce it to emergency_threshold
        if threshold_cf >= emergency_threshold:
            threshold_cf = emergency_threshold

        

        #print statement (save only good results)
        # if accuracy >= threshold_cf:
        if accuracy >= threshold_cf: #forget about threshold for a while

            #identifierfor saving the models
            identifier+= 1
            
            #confusion matrix
            #create a list of ordered labels according to the number of classes
            labels = list(range(1,len(Y_train.iloc[:,target].unique())+1))

            #get name of the interval
            column_name = Y_test.iloc[:,target].name
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

            #save confusion matrix
            cm = confusion_matrix(Y_test.iloc[:,target], predictions, labels=labels)

            #create a dataframe to be saved 
            cmtx = pd.DataFrame(
                cm, 
                index=index, 
                columns=column
            )

            print("\nAccuracy: %s" %round(accuracy,4))
            print("Accuracy dummy was: %s" %round(accuracy_dummy,4))
            print("Accuracy of this setting is greater than threshold (%s). Print confusion matrix! \n %s" %(round(threshold_cf,4),cm))
            
            #for proper naming of the files, some transformations are needed
            #prepros
            preprocessing_str = ""
            for p in preprocessing:
                preprocessing_str += ("_" + p.__class__.__name__)
           
            #cv
            my_dict = cv.__dict__
            default = cv.__class__().__dict__
            cv_str = ""
            for i in my_dict:
                if my_dict[i] == default[i]:
                    continue
                else:
                    cv_str += ("cv_" + i + "_" + str(my_dict[i]))
            if cv_str == "":
                cv_str = "cv_default"

            #labels string
            #remove spaces       
            label_name_save = get_digits(column_name[:32]) + column_name[index_first:].replace(" ", "").replace("'", "")

            # path to save cf matrix
            #normalized
            path_visualization = "results/%s/cm/%s_%s_%s_%s_%s_%s_normalized.jpg" %( symbol[:-4], identifier, label_name_save,  model.__class__.__name__, preprocessing_str, cv_str, model_params_str)           
            plot_confusion_matrix(cm, plot_labels, path_visualization, display=False, normalize=True,
             title="Target: %s \nSymbol: %s\nModel: %s\nPreprocessing: %s\n\nAccuracy: %s \nDummy accuracy %s" %(label_name_save, symbol[:-4],
              model.__class__.__name__, preprocessing_str,
             round(accuracy, 4), round(accuracy_dummy, 4)))

            #not normalized
            path_visualization = "results/%s/cm/%s_%s_%s_%s_%s_%s.jpg" %( symbol[:-4],identifier,label_name_save, model.__class__.__name__, preprocessing_str, cv_str,  model_params_str) 
            plot_confusion_matrix(cm, plot_labels, path_visualization, display=False, normalize=False,
             title="Target: %s \nSymbol: %s\nModel: %s\nPreprocessing: %s\n\nAccuracy: %s \nDummy accuracy %s" %(label_name_save, symbol[:-4],
              model.__class__.__name__, preprocessing_str,
             round(accuracy, 4), round(accuracy_dummy, 4)))

            #save (non-normalized) dataframe confusion matrix
            cmtx.to_csv("results/%s/cm/%s_%s_%s_%s_%s_%s.csv" %(symbol[:-4], identifier,  label_name_save, model.__class__.__name__, preprocessing_str, cv_str, model_params_str),
            sep=";", decimal=",",
            #index True in this case to have the labels obtained before
            index=True)

            #also save parameters from grid search when accuracy greater than threshold
            cv_results = pd.DataFrame(search.cv_results_)
            cv_results.to_csv("results/%s/cv/%s_%s_%s_%s_%s_%s.csv" %( symbol[:-4], identifier, label_name_save,  model.__class__.__name__, preprocessing_str, cv_str, model_params_str),
             sep=";", decimal=",", index=False)

            #save also best estimator
            best_estimator = search.best_estimator_
            dump(best_estimator,
             "results/%s/models/%s_%s_%s_%s_%s_%s.joblib" %(symbol[:-4], identifier, label_name_save,  model.__class__.__name__, preprocessing_str, cv_str, model_params_str))
            print("Model_saved!")

            #load model and check accuracy
            # clf = load("results/%s/models/%s_%s_%s_%s_%s_%s.joblib" %(symbol[:-4], identifier, label_name_save,  model.__class__.__name__, preprocessing_str, cv_str, model_params_str))
            # predictions = clf.predict(X_test)
            # print("Accuracy before: ", accuracy)
            # print("Accuracy after: ", accuracy_score(Y_test.iloc[:,target], predictions))
        
        else:
            print("Accuracy %s is under threshold %s" %(accuracy,threshold_cf))

    #create a dictionary to append row to dataframe
    row = {}

    #append basic model columns 
    row[model_columns[0]] = symbol
    row[model_columns[1]] = model.__class__.__name__
    row[model_columns[2]] = preprocessing
    row[model_columns[3]] = cv
    row[model_columns[4]] = model_params

    #for indexing evaluation list
    count = 0
    for target in targets:
        for evaluation in evaluation_columns:
            #make the name of the target cleaner
            m = re.search(r"\d", target).start()
            row[evaluation+"_"+target[m:]] = evaluation_list[count]
            count += 1


    #add also training and prediction times
    row["training_time"] = train_total
    row["prediction_time"] = pred_total
    row["round"] = exp_round


    #append row to dataset
    results = results.append(row, ignore_index=True)

    return results 


#################################
# MULTIPLE TRAINING FOR A STOCK #
#################################


"""
    TODO 
"""

# loop over cross validations, preprocessing pipelines and models
def multiple_training(cvs, preprocessings, classifiers, params, X_train, Y_train, X_test, Y_test, symbol,
     results, model_columns, targets, evaluation_columns, threshold, emergency_threshold, intervals, shiftDays, symbols, exp_round
    ):

    print("\n\n\n\nStarting training...")
    count = 0

    for cv in cvs:
        for preprocessing in preprocessings:
            for classifier in classifiers:
                print("\n\n\n")
                print("Model: %s" %count)
                print(cv)
                print(preprocessing)
                print(classifier)                
                results = grid_search_model(cv, preprocessing, classifier, params[count], X_train, Y_train, X_test, Y_test, symbol,
                 results, model_columns, targets, evaluation_columns, threshold, emergency_threshold, exp_round, verbose=0)
                count += 1

    print("\n\n\n\n\nIntervals to target: ", len(intervals))
    print("Days to predict: ", len(shiftDays))
    print("Number of models: ", count)
    print("Therefore, total grid searches (per stock:) ", count*len(intervals)*len(shiftDays))
    total_searches = count*len(intervals)*len(shiftDays)*len(symbols)
    print("\nTotal Cross Validations searches to be trained: ", total_searches)
    total_models = 5*(total_searches*(1/2)) + 10*(total_searches*(1/2))
    print("Total models to be trained per split in the grid search CV: ", total_models)
    gpus = multiprocessing.cpu_count()
    print("\nTotal cores", gpus)
    print("Total models per core: ", total_models/gpus)
    print("\nTotal models to be trained if there are %s combinations of hyperparams in the grid search : " %10, total_models*10)

    return results