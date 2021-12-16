import os
import glob
import shutil

folders = []

for folder in os.listdir("results/"):
    folders.append(folder)

#remove all folders
for folder in folders:
    if folder == "results.csv":
        continue
    shutil.rmtree("results/%s" %folder)

#create new ones
for folder in folders:
    if folder == "results.csv":
        continue
    os.mkdir("results/%s" %folder)
    os.mkdir("results/%s/cm" %folder)
    os.mkdir("results/%s/cv" %folder)
    os.mkdir("results/%s/models" %folder)
    