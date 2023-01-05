"""Stacked barplot showcasing the number of pos, neg, netural responses for each topic"""
import numpy as np
from matplotlib import pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import statistical_analysis as sa
import statistics
import pandas as pd


def count_sentiment_pos_percentage(first: str, second: str, third: str):
    """Counts the percentage of positive responses given a data file"""
    sid_obj = SentimentIntensityAnalyzer()
    analyze2019 = sa.clean_likert_frq_open_file(first)
    analyze2020 = sa.clean_likert_frq_open_file(second)
    analyze2021 = sa.clean_likert_frq_open_file(third)
    pos_count = 0
    neg_count = 0
    neu_count = 0

    for item in analyze2019:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)

            if sentiment_dict['compound'] >= 0.05:
                pos_count += 1

            elif sentiment_dict['compound'] <= - 0.05:
                neg_count += 1
            else:
                neu_count += 1
    for item in analyze2020:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)

            if sentiment_dict['compound'] >= 0.05:
                pos_count += 1

            elif sentiment_dict['compound'] <= - 0.05:
                neg_count += 1
            else:
                neu_count += 1
    for item in analyze2021:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)

            if sentiment_dict['compound'] >= 0.05:
                pos_count += 1

            elif sentiment_dict['compound'] <= - 0.05:
                neg_count += 1
            else:
                neu_count += 1

    total = pos_count + neg_count + neu_count
    return (pos_count / total) * 100


def count_sentiment_neg_percentage(first: str, second: str, third: str):
    """Counts the percentage of negative responses given a data file"""
    sid_obj = SentimentIntensityAnalyzer()
    analyze2019 = sa.clean_likert_frq_open_file(first)
    analyze2020 = sa.clean_likert_frq_open_file(second)
    analyze2021 = sa.clean_likert_frq_open_file(third)
    pos_count = 0
    neg_count = 0
    neu_count = 0

    for item in analyze2019:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)

            if sentiment_dict['compound'] >= 0.05:
                pos_count += 1

            elif sentiment_dict['compound'] <= - 0.05:
                neg_count += 1
            else:
                neu_count += 1
    for item in analyze2020:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)

            if sentiment_dict['compound'] >= 0.05:
                pos_count += 1

            elif sentiment_dict['compound'] <= - 0.05:
                neg_count += 1
            else:
                neu_count += 1
    for item in analyze2021:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)

            if sentiment_dict['compound'] >= 0.05:
                pos_count += 1

            elif sentiment_dict['compound'] <= - 0.05:
                neg_count += 1
            else:
                neu_count += 1

    total = pos_count + neg_count + neu_count
    return (neg_count / total) * 100


def count_sentiment_neu_percentage(first: str, second: str, third: str):
    """Counts the percentage of neutral responses given a data file"""
    sid_obj = SentimentIntensityAnalyzer()
    analyze2019 = sa.clean_likert_frq_open_file(first)
    analyze2020 = sa.clean_likert_frq_open_file(second)
    analyze2021 = sa.clean_likert_frq_open_file(third)
    pos_count = 0
    neg_count = 0
    neu_count = 0

    for item in analyze2019:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)

            if sentiment_dict['compound'] >= 0.05:
                pos_count += 1

            elif sentiment_dict['compound'] <= - 0.05:
                neg_count += 1
            else:
                neu_count += 1
    for item in analyze2020:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)

            if sentiment_dict['compound'] >= 0.05:
                pos_count += 1

            elif sentiment_dict['compound'] <= - 0.05:
                neg_count += 1
            else:
                neu_count += 1
    for item in analyze2021:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)

            if sentiment_dict['compound'] >= 0.05:
                pos_count += 1

            elif sentiment_dict['compound'] <= - 0.05:
                neg_count += 1
            else:
                neu_count += 1

    total = pos_count + neg_count + neu_count
    return (neu_count / total) * 100


def stacked_barplot():
    positive = [count_sentiment_pos_percentage('winter2019_understanding.csv', 'winter2020_understanding.csv',
                                               'winter2021_understanding.csv'),
                count_sentiment_pos_percentage('winter2019_skill.csv', 'winter2020_skill.csv',
                                               'winter2021_skill.csv'),
                count_sentiment_pos_percentage('winter2019_attitude.csv', 'winter2020_attitude.csv',
                                               'winter2021_attitude.csv'),
                count_sentiment_pos_percentage('winter2019_integrate.csv', 'winter2020_integrate.csv',
                                               'winter2021_integrate.csv'),
                count_sentiment_pos_percentage('winter2019_overall.csv', 'winter2020_overall.csv',
                                               'winter2021_overall.csv'),
                count_sentiment_pos_percentage('winter2019_activities.csv', 'winter2020_activities.csv',
                                               'winter2021_activities.csv'),
                count_sentiment_pos_percentage('winter2019_assignments.csv', 'winter2020_assignments.csv',
                                               'winter2021_assignments.csv'),
                count_sentiment_pos_percentage('winter2019_resources.csv', 'winter2020_resources.csv',
                                               'winter2021_resources.csv'),
                count_sentiment_pos_percentage('winter2019_info.csv', 'winter2020_info.csv',
                                               'winter2021_info.csv'),
                count_sentiment_pos_percentage('winter2019_support.csv', 'winter2020_support.csv',
                                               'winter2021_support.csv'),
                count_sentiment_pos_percentage('winter2019_improvement.csv', 'winter2020_improvement.csv',
                                               'winter2021_improvement.csv')]

    negative = [count_sentiment_neg_percentage('winter2019_understanding.csv', 'winter2020_understanding.csv',
                                               'winter2021_understanding.csv'),
                count_sentiment_neg_percentage('winter2019_skill.csv', 'winter2020_skill.csv',
                                               'winter2021_skill.csv'),
                count_sentiment_neg_percentage('winter2019_attitude.csv', 'winter2020_attitude.csv',
                                               'winter2021_attitude.csv'),
                count_sentiment_neg_percentage('winter2019_integrate.csv', 'winter2020_integrate.csv',
                                               'winter2021_integrate.csv'),
                count_sentiment_neg_percentage('winter2019_overall.csv', 'winter2020_overall.csv',
                                               'winter2021_overall.csv'),
                count_sentiment_neg_percentage('winter2019_activities.csv', 'winter2020_activities.csv',
                                               'winter2021_activities.csv'),
                count_sentiment_neg_percentage('winter2019_assignments.csv', 'winter2020_assignments.csv',
                                               'winter2021_assignments.csv'),
                count_sentiment_neg_percentage('winter2019_resources.csv', 'winter2020_resources.csv',
                                               'winter2021_resources.csv'),
                count_sentiment_neg_percentage('winter2019_info.csv', 'winter2020_info.csv',
                                               'winter2021_info.csv'),
                count_sentiment_neg_percentage('winter2019_support.csv', 'winter2020_support.csv',
                                               'winter2021_support.csv'),
                count_sentiment_neg_percentage('winter2019_improvement.csv', 'winter2020_improvement.csv',
                                               'winter2021_improvement.csv')]

    neutral = [count_sentiment_neu_percentage('winter2019_understanding.csv', 'winter2020_understanding.csv',
                                              'winter2021_understanding.csv'),
               count_sentiment_neu_percentage('winter2019_skill.csv', 'winter2020_skill.csv',
                                              'winter2021_skill.csv'),
               count_sentiment_neu_percentage('winter2019_attitude.csv', 'winter2020_attitude.csv',
                                              'winter2021_attitude.csv'),
               count_sentiment_neu_percentage('winter2019_integrate.csv', 'winter2020_integrate.csv',
                                              'winter2021_integrate.csv'),
               count_sentiment_neu_percentage('winter2019_overall.csv', 'winter2020_overall.csv',
                                              'winter2021_overall.csv'),
               count_sentiment_neu_percentage('winter2019_activities.csv', 'winter2020_activities.csv',
                                              'winter2021_activities.csv'),
               count_sentiment_neu_percentage('winter2019_assignments.csv', 'winter2020_assignments.csv',
                                              'winter2021_assignments.csv'),
               count_sentiment_neu_percentage('winter2019_resources.csv', 'winter2020_resources.csv',
                                              'winter2021_resources.csv'),
               count_sentiment_neu_percentage('winter2019_info.csv', 'winter2020_info.csv',
                                              'winter2021_info.csv'),
               count_sentiment_neu_percentage('winter2019_support.csv', 'winter2020_support.csv',
                                              'winter2021_support.csv'),
               count_sentiment_neu_percentage('winter2019_improvement.csv', 'winter2020_improvement.csv',
                                              'winter2021_improvement.csv')]

    index = ['understanding', 'skill', 'attitude', 'integrate', 'overall', 'activities', 'assignments', 'resources',
             'info', 'support', 'improvement']

    df = pd.DataFrame({'positive': positive, 'neutral': neutral, 'negative': negative}, index=index)
    ax = df.plot.barh(figsize=(20, 10), stacked=True, color=['#FDBD59', '#C9E265', '#82C0D9'])
    group = ['understanding', 'skill', 'attitude', 'integrate', 'overall', 'activities', 'assignments', 'resources',
             'info', 'support', 'improvement']
    ax.set_yticklabels(group, size=15)
    ax.set_xticklabels(ax.get_xticks(), size=15)
    ax.set_xlabel('percentage', fontsize=20)
    ax.set_ylabel('section',fontsize=20)
    ax.invert_yaxis()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=14)
