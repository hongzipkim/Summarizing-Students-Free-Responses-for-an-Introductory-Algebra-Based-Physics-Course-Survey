"""This file contains the code to calculate the F1 score for VADER sentiment vs. human sentiment labelling."""
import csv
import random

import pandas as pd

import statistical_analysis as sta
import sentiment_analysis as senta

from sklearn.metrics import f1_score


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

    random_list = random.choices(huge_list, k=300)
    df = pd.DataFrame(random_list)
    return df


def f1_vader(file1: str, file2: str):
    """Calculates the Kappa coefficient between VADER vs. human annotator
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

    # calculate Cohen's Kappa
    return f1_score(listy, rater2, average='micro')
