"""
This file contains the code to calculate the F1 score for VADER sentiment vs. human sentiment labelling,
and to calculate the Inter-Annotator Agreement (IAA) using the kappa coefficient"""
import csv
import random
import pandas as pd

import statistical_analysis as sta
import sentiment_analysis as senta

from sklearn.metrics import cohen_kappa_score, f1_score


def entire_list(first: str, second: str, third: str):
    """Returns an exhaustive list of all 3 years of data combined."""
    winter2019 = sta.clean_likert_frq_open_file(first)
    winter2020 = sta.clean_likert_frq_open_file(second)
    winter2021 = sta.clean_likert_frq_open_file(third)
    listy = []
    for item in winter2019:
        for sentence in item[1]:
            listy.append(sentence)
    for item in winter2020:
        for sentence in item[1]:
            listy.append(sentence)
    for item in winter2021:
        for sentence in item[1]:
            listy.append(sentence)
    return listy


def extracting_sample():
    """Returns a data frame of a random sample of 300 student responses."""
    topic1 = entire_list('winter2019_understanding.csv', 'winter2020_understanding.csv',
                         'winter2021_understanding.csv')
    topic2 = entire_list('winter2019_skill.csv', 'winter2020_skill.csv',
                         'winter2021_skill.csv')
    topic3 = entire_list('winter2019_attitude.csv', 'winter2020_attitude.csv',
                         'winter2021_attitude.csv')
    topic4 = entire_list('winter2019_integrate.csv', 'winter2020_integrate.csv',
                         'winter2021_integrate.csv')
    topic5 = entire_list('winter2019_overall.csv', 'winter2020_overall.csv',
                         'winter2021_overall.csv')
    topic6 = entire_list('winter2019_activities.csv', 'winter2020_activities.csv',
                         'winter2021_activities.csv')
    topic7 = entire_list('winter2019_assignments.csv', 'winter2020_assignments.csv',
                         'winter2021_assignments.csv')
    topic8 = entire_list('winter2019_resources.csv', 'winter2020_resources.csv',
                         'winter2021_resources.csv')
    topic9 = entire_list('winter2019_info.csv', 'winter2020_info.csv',
                         'winter2021_info.csv')
    topic10 = entire_list('winter2019_support.csv', 'winter2020_support.csv',
                          'winter2021_support.csv')
    topic11 = entire_list('winter2019_improvement.csv', 'winter2020_improvement.csv', 'winter2021_improvement.csv')

    huge_list = topic1 + topic2 + topic3 + topic4 + topic5 + topic6 + topic7 + topic8 + topic9 + \
                topic10 + topic11

    random_list = random.choices(huge_list, k=800)
    df = pd.DataFrame(random_list)
    return df


def f1_vader(file1: str, file2: str):
    """Calculates the F1 score between VADER vs. human annotator
    file1 consists of VADER sentiment
    file2 consists of human annotator"""

    # define array of ratings for both raters
    listy = []
    rater1 = senta.count_sentiment_for_f1(file1)
    for item in rater1:
        if item == 'Positive':
            listy.append(1)
        elif item == 'Neutral':
            listy.append(0)
        elif item == 'Negative':
            listy.append(-1)
    # print(listy)

    lst = []
    lista = []
    with open(file2) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            lst.append(row)
        # print(lst)
        for item in lst:
            if item == ['n']:
                lista.append(0)
            elif item == ['p']:
                lista.append(1)
            elif item == ['ne']:
                lista.append(-1)
    # print(lista)
    rater2 = lista

    # calculate F1 score
    # note: change 'micro' to 'macro' if calculating the macro F1 score
    return f1_score(listy, rater2, average='micro')

def kappa_iaa(file1: str, file2: str):
    """Calculates the Kappa coefficient between human annotator vs. human annotator"""

    # define array of ratings for both raters
    lst1 = []
    lista1 = []
    with open(file1) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            lst1.append(row)
        print(lst1)
        for item in lst1:
            if item == ['n']:
                lista1.append(0)
            elif item == ['p']:
                lista1.append(1)
            elif item == ['ne']:
                lista1.append(-1)
    print(lista1)
    rater1 = lista1

    lst2 = []
    lista2 = []
    with open(file2) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            lst2.append(row)
        print(lst2)
        for item in lst2:
            if item == ['n']:
                lista2.append(0)
            elif item == ['p']:
                lista2.append(1)
            elif item == ['ne']:
                lista2.append(-1)
    print(lista2)
    rater2 = lista2

    # calculate Cohen's Kappa score
    return cohen_kappa_score(rater1, rater2)
