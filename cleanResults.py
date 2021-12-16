#script used to "clean results dataframe. It selects only the necessary columns
#for my analysis as exports this information into a csv file and excel file
#called results_clean (csv as a backup and excel for editing)

import pandas as pd

def clean_results():

    results = pd.read_csv("results/results.csv", sep=";", decimal=",")

    results.iloc[:,[0,1,2,3,4,
    5,9,10,11,
    15,
    25,29,30,31,
    35,
    45,49,50,51,
    55,
    65,66,67]].to_csv("results/results_clean.csv", sep=";", index=False, decimal=",")

    results.iloc[:,[0,1,2,3,4,
    5,9,10,11,
    15,
    25,29,30,31,
    35,
    45,49,50,51,
    55,
    65,66,67]].to_excel("results/results_clean.xlsx", index=False)

