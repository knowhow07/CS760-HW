#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 20:44:44 2023

@author: nuohaoliu
"""

# k-nearest neighbors on the Iris Flowers Dataset
import csv
import math
from random import seed
from random import randrange
from csv import reader
import sys
import time
from datetime import timedelta
start_time = time.monotonic()


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])





# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return(predictions)

# # Split a dataset into k folds
# def cross_validation_split(dataset, n_folds):
#  dataset_split = list()
#  dataset_copy = list(dataset)
#  fold_size = int(len(dataset) / n_folds)
#  for _ in range(n_folds):
#      fold = list()
#      while len(fold) < fold_size:
#          index = randrange(len(dataset_copy))
#          fold.append(dataset_copy.pop(index))
#      dataset_split.append(fold)
#  return dataset_split

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for n in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = 0
            # index = randrange(len(dataset_copy))
                # pop = dataset_copy.pop(index)
            fold.append(dataset_copy.pop(index))
                # index +=1
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage

# # Calculate accuracy percentage
# def accuracy_metric(actual, predicted):
#  correct = 0
#  for i in range(len(actual)):
#      if actual[i] == predicted[i]:
#          correct += 1
#  return correct / float(len(actual)) * 100.0

def accuracy_metric(actual, predicted):
    correct = 0
    pos = 0
    neg = 0
    true_pos = 0
    false_neg = 0
    false_pos = 0
    true_neg = 0
    for i in range(len(actual)):
        if actual[i] == 1:
            pos += 1
            if actual[i] == predicted[i]:
                true_pos += 1
                correct += 1
        else:
            neg += 1
            if actual[i] == predicted[i]:
                true_neg += 1
                correct += 1
    false_pos = neg - true_neg
    false_neg = pos - true_pos
    accuracy = correct / float(len(actual)) * 100.0
    if (true_pos + false_pos) == 0:
        precision = 0.0
        print('precisition divide by 0')
    else:
        precision =  true_pos/ (true_pos + false_pos) * 100.0
        
    if pos == 0:
        recall = 0.0
        print('recall divide by 0')
    else:
        recall = true_pos / pos * 100.0
    return accuracy, precision, recall

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    accus = list()
    precs = list()
    recalls = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy, precision, recall = accuracy_metric(actual, predicted)
        accus.append(accuracy)
        precs.append(precision)
        recalls.append(recall)
    return accus,precs,recalls

# Test the kNN on the Iris Flowers dataset
seed(1)
# filename = 'data/iris.csv'
filename = 'emails2.csv'
dataset = load_csv(filename)
name = dataset[0]
dataset.remove(dataset[0])
# exclude = ['Email']
# res = [ele for ele in dataset if all(ch not in ele for ch in exclude)]
# for word in list(dataset):  # iterating on a copy since removing will mess things up
#     if word in ['Email']:
#         dataset.remove(word)

for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
    
for row in dataset:
    row.pop(0)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
num_neighbors = 1
accus,precs,recalls = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
print('num_neighbors=',num_neighbors)
print('accus: %s \n, precs: %s \n, recalls: %s'  % (accus,precs,recalls))
print('Mean Accuracy: %.3f%%, Mean Precision: %.3f%%, Mean Recall: %.3f%%' 
      % (sum(accus)/float(len(accus)),sum(precs)/float(len(precs)),sum(recalls)/float(len(recalls))))
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time)) 