"""
This file contains code necessary to produce a confusion matrix.
"""
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def open_file(file: str):
    """ Opens the SALG data file and creates a nested list tuple with the following format: (text, sentiment)."""
    listy = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            text = ''
            sentiment = ''
            for cell in row:
                if cell == 'ne' or cell == 'p' or cell == 'n':
                    sentiment = cell
                else:
                    text = cell
            if sentiment != '':
                listy.append((text, sentiment))
    return listy


def analyze_confusion_matrix():
    """Analyzes the given input file, and produces a 3 x 3 confusion matrix."""
    analyze = open_file('naive_bayes_testset.csv')  # input any csv file with columns: sentence, sentiment score
    tp = 0
    fp1 = 0
    fp2 = 0
    fng1 = 0
    tng = 0
    fng2 = 0
    fnt1 = 0
    fnt2 = 0
    tnt = 0

    sid_obj = SentimentIntensityAnalyzer()
    listy = []
    for item in analyze:
        sentiment_dict = sid_obj.polarity_scores(item[0])
        if sentiment_dict['compound'] >= 0.05:
            listy.append('p')

        elif sentiment_dict['compound'] <= - 0.05:
            listy.append('ne')
        else:
            listy.append('n')

    print(listy)

    for (i, j) in zip(listy, analyze):
        if i == 'p' and j[1] == 'p':
            tp += 1
        elif i == 'p' and j[1] == 'ne':
            fng1 += 1
        elif i == 'p' and j[1] == 'n':
            fnt1 += 1
        elif i == 'ne' and j[1] == 'p':
            fp1 += 1
        elif i == 'ne' and j[1] == 'ne':
            tng += 1
        elif i == 'ne' and j[1] == 'n':
            fnt2 += 1
        elif i == 'n' and j[1] == 'p':
            fp2 += 1
        elif i == 'n' and j[1] == 'ne':
            fng2 += 1
        elif i == 'n' and j[1] == 'n':
            tnt += 1

    print('Positive Column', tp, fp1, fp2)
    print('Negative Column', fng1, tng, fng2)
    print('Netural Column', fnt1, fnt2, tnt)
