"""Code to compare the sentiment of extreme vs. remaining responses"""
import nltk
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import statistical_analysis as sa
import sentiment_analysis as senta
import statistics
import pandas as pd
import seaborn as sns
import numpy as np


def grand_list(first: str, second: str, third: str):
    analyze2019 = sa.clean_likert_frq_open_file(first)
    analyze2020 = sa.clean_likert_frq_open_file(second)
    analyze2021 = sa.clean_likert_frq_open_file(third)
    extreme = []
    remaining = []
    for item in analyze2019:
        for sentence in item[1]:
            a_list = nltk.tokenize.sent_tokenize(sentence)
            if len(a_list) > 1:
                if item not in extreme:
                    extreme.append(item)
            else:
                if item not in extreme and item not in remaining:
                    remaining.append(item)
    for item in analyze2020:
        for sentence in item[1]:
            a_list = nltk.tokenize.sent_tokenize(sentence)
            if len(a_list) > 1:
                if item not in extreme:
                    extreme.append(item)
                else:
                    if item not in extreme and item not in remaining:
                        remaining.append(item)
    for item in analyze2021:
        for sentence in item[1]:
            a_list = nltk.tokenize.sent_tokenize(sentence)
            if len(a_list) > 1:
                if item not in extreme:
                    extreme.append(item)
                else:
                    if item not in extreme and item not in remaining:
                        remaining.append(item)
    return [extreme, remaining]


def ind_sent(function: list[tuple]):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_list = []
    for item in function:
        listy = []
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)
            listy.append(sentiment_dict['compound'])
        sentiment_list.append(listy)
    return sentiment_list


def nested_list_of_likert(function: list[tuple]):
    likert = []
    for item in function:
        nested = []
        for i in item[0]:
            if int(i) == 1:
                nested.append(-1)
            elif int(i) == 2:
                nested.append(-0.5)
            elif int(i) == 3:
                nested.append(0)
            elif int(i) == 4:
                nested.append(0.5)
            elif int(i) == 5:
                nested.append(1)
        likert.append(nested)
    return likert


def get_the_total_data(title: str, function):
    likey = nested_list_of_likert(function)
    senty = ind_sent(function)
    likert = []
    sentiment = []
    group = []
    for sent in senty:
        for each_sent in sent:
            for like in likey:
                for each in like:
                    likert.append(each)
                    group.append(title)
                    sentiment.append(each_sent)
    total_data = {'length of response': group, 'likert': likert, 'sentiment': sentiment}
    return total_data


def new_boxplot():
    topic1 = get_the_total_data('long', grand_list('winter2019_understanding.csv', 'winter2020_understanding.csv',
                                                   'winter2021_understanding.csv')[0])
    topic2 = get_the_total_data('long',
                                grand_list('winter2019_skill.csv', 'winter2020_skill.csv', 'winter2021_skill.csv')[0])
    topic3 = get_the_total_data('long', grand_list('winter2019_attitude.csv', 'winter2020_attitude.csv',
                                                   'winter2021_attitude.csv')[0])
    topic4 = get_the_total_data('long', grand_list('winter2019_integrate.csv', 'winter2020_integrate.csv',
                                                   'winter2021_integrate.csv')[0])
    topic5 = get_the_total_data('long', grand_list('winter2019_overall.csv', 'winter2020_overall.csv',
                                                   'winter2021_overall.csv')[0])
    topic6 = get_the_total_data('long', grand_list('winter2019_activities.csv', 'winter2020_activities.csv',
                                                   'winter2021_activities.csv')[0])
    topic7 = get_the_total_data('long', grand_list('winter2019_assignments.csv', 'winter2020_assignments.csv',
                                                   'winter2021_assignments.csv')[0])
    topic8 = get_the_total_data('long', grand_list('winter2019_resources.csv', 'winter2020_resources.csv',
                                                   'winter2021_resources.csv')[0])
    topic9 = get_the_total_data('long', grand_list('winter2019_info.csv', 'winter2020_info.csv',
                                                   'winter2021_info.csv')[0])
    topic10 = get_the_total_data('long', grand_list('winter2019_support.csv', 'winter2020_support.csv',
                                                    'winter2021_support.csv')[0])
    topic11 = get_the_total_data('short',
                                 grand_list('winter2019_understanding.csv', 'winter2020_understanding.csv',
                                            'winter2021_understanding.csv')[1])
    topic12 = get_the_total_data('short',
                                 grand_list('winter2019_skill.csv', 'winter2020_skill.csv', 'winter2021_skill.csv')[1])
    topic13 = get_the_total_data('short', grand_list('winter2019_attitude.csv', 'winter2020_attitude.csv',
                                                     'winter2021_attitude.csv')[1])
    topic14 = get_the_total_data('short', grand_list('winter2019_integrate.csv', 'winter2020_integrate.csv',
                                                     'winter2021_integrate.csv')[1])
    topic15 = get_the_total_data('short', grand_list('winter2019_overall.csv', 'winter2020_overall.csv',
                                                     'winter2021_overall.csv')[1])
    topic16 = get_the_total_data('short', grand_list('winter2019_activities.csv', 'winter2020_activities.csv',
                                                     'winter2021_activities.csv')[1])
    topic17 = get_the_total_data('short', grand_list('winter2019_assignments.csv', 'winter2020_assignments.csv',
                                                     'winter2021_assignments.csv')[1])
    topic18 = get_the_total_data('short', grand_list('winter2019_resources.csv', 'winter2020_resources.csv',
                                                     'winter2021_resources.csv')[1])
    topic19 = get_the_total_data('short', grand_list('winter2019_info.csv', 'winter2020_info.csv',
                                                     'winter2021_info.csv')[1])
    topic20 = get_the_total_data('short', grand_list('winter2019_support.csv', 'winter2020_support.csv',
                                                     'winter2021_support.csv')[1])
    d0 = merge_dictionaries(topic1, topic2)
    d1 = merge_dictionaries(d0, topic3)
    d2 = merge_dictionaries(d1, topic4)
    d3 = merge_dictionaries(d2, topic5)
    d4 = merge_dictionaries(d3, topic6)
    d5 = merge_dictionaries(d4, topic7)
    d6 = merge_dictionaries(d5, topic8)
    d7 = merge_dictionaries(d6, topic9)
    d8 = merge_dictionaries(d7, topic10)
    d9 = merge_dictionaries(d8, topic11)
    d10 = merge_dictionaries(d9, topic12)
    d11 = merge_dictionaries(d10, topic13)
    d12 = merge_dictionaries(d11, topic14)
    d13 = merge_dictionaries(d12, topic15)
    d14 = merge_dictionaries(d13, topic16)
    d15 = merge_dictionaries(d14, topic17)
    d16 = merge_dictionaries(d15, topic18)
    d17 = merge_dictionaries(d16, topic19)
    d18 = merge_dictionaries(d17, topic20)
    df = pd.DataFrame(data=d18)
    print(df)
    dd = pd.melt(df, id_vars=['length of response'], value_vars=['sentiment', 'likert'], var_name='type')
    fig = plt.figure(figsize=(20, 20))

    rgb = [(255, 204, 153),
           (128, 229, 255)]
    colors = [tuple(t / 255 for t in x) for x in rgb]
    b = sns.boxplot(x='length of response', y='value', data=dd, hue='type', medianprops=dict(color="red", alpha=0.9),
                    palette=colors)
    b.set_yticklabels(b.get_yticks(), size=15)
    b.set_xticklabels(['long', 'short'], size=15)
    b.set_xlabel("length of response", fontsize=20)
    b.set_ylabel("score", fontsize=20)
    adjust_box_widths(fig, 0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=10)


def get_data():
    topic1 = get_the_total_data('long', grand_list('winter2019_understanding.csv', 'winter2020_understanding.csv',
                                                   'winter2021_understanding.csv')[0])
    topic2 = get_the_total_data('long',
                                grand_list('winter2019_skill.csv', 'winter2020_skill.csv', 'winter2021_skill.csv')[0])
    topic3 = get_the_total_data('long', grand_list('winter2019_attitude.csv', 'winter2020_attitude.csv',
                                                   'winter2021_attitude.csv')[0])
    topic4 = get_the_total_data('long', grand_list('winter2019_integrate.csv', 'winter2020_integrate.csv',
                                                   'winter2021_integrate.csv')[0])
    topic5 = get_the_total_data('long', grand_list('winter2019_overall.csv', 'winter2020_overall.csv',
                                                   'winter2021_overall.csv')[0])
    topic6 = get_the_total_data('long', grand_list('winter2019_activities.csv', 'winter2020_activities.csv',
                                                   'winter2021_activities.csv')[0])
    topic7 = get_the_total_data('long', grand_list('winter2019_assignments.csv', 'winter2020_assignments.csv',
                                                   'winter2021_assignments.csv')[0])
    topic8 = get_the_total_data('long', grand_list('winter2019_resources.csv', 'winter2020_resources.csv',
                                                   'winter2021_resources.csv')[0])
    topic9 = get_the_total_data('long', grand_list('winter2019_info.csv', 'winter2020_info.csv',
                                                   'winter2021_info.csv')[0])
    topic10 = get_the_total_data('long', grand_list('winter2019_support.csv', 'winter2020_support.csv',
                                                    'winter2021_support.csv')[0])
    topic11 = get_the_total_data('short',
                                 grand_list('winter2019_understanding.csv', 'winter2020_understanding.csv',
                                            'winter2021_understanding.csv')[1])
    topic12 = get_the_total_data('short',
                                 grand_list('winter2019_skill.csv', 'winter2020_skill.csv', 'winter2021_skill.csv')[1])
    topic13 = get_the_total_data('short', grand_list('winter2019_attitude.csv', 'winter2020_attitude.csv',
                                                     'winter2021_attitude.csv')[1])
    topic14 = get_the_total_data('short', grand_list('winter2019_integrate.csv', 'winter2020_integrate.csv',
                                                     'winter2021_integrate.csv')[1])
    topic15 = get_the_total_data('short', grand_list('winter2019_overall.csv', 'winter2020_overall.csv',
                                                     'winter2021_overall.csv')[1])
    topic16 = get_the_total_data('short', grand_list('winter2019_activities.csv', 'winter2020_activities.csv',
                                                     'winter2021_activities.csv')[1])
    topic17 = get_the_total_data('short', grand_list('winter2019_assignments.csv', 'winter2020_assignments.csv',
                                                     'winter2021_assignments.csv')[1])
    topic18 = get_the_total_data('short', grand_list('winter2019_resources.csv', 'winter2020_resources.csv',
                                                     'winter2021_resources.csv')[1])
    topic19 = get_the_total_data('short', grand_list('winter2019_info.csv', 'winter2020_info.csv',
                                                     'winter2021_info.csv')[1])
    topic20 = get_the_total_data('short', grand_list('winter2019_support.csv', 'winter2020_support.csv',
                                                     'winter2021_support.csv')[1])
    d0 = merge_dictionaries(topic1, topic2)
    d1 = merge_dictionaries(d0, topic3)
    d2 = merge_dictionaries(d1, topic4)
    d3 = merge_dictionaries(d2, topic5)
    d4 = merge_dictionaries(d3, topic6)
    d5 = merge_dictionaries(d4, topic7)
    d6 = merge_dictionaries(d5, topic8)
    d7 = merge_dictionaries(d6, topic9)
    d8 = merge_dictionaries(d7, topic10)
    df1 = pd.DataFrame(data=d8)
    d9 = merge_dictionaries(topic11, topic12)
    d10 = merge_dictionaries(d9, topic13)
    d11 = merge_dictionaries(d10, topic14)
    d12 = merge_dictionaries(d11, topic15)
    d13 = merge_dictionaries(d12, topic16)
    d14 = merge_dictionaries(d13, topic17)
    d15 = merge_dictionaries(d14, topic18)
    d16 = merge_dictionaries(d15, topic19)
    d17 = merge_dictionaries(d16, topic20)
    df2 = pd.DataFrame(data=d17)
    print(df1.describe().round(3))
    print(df2.describe().round(3))


def merge_dictionaries(dict1, dict2):
    merged_dictionary = {}

    for key in dict1:
        if key in dict2:
            new_value = dict1[key] + dict2[key]
        else:
            new_value = dict1[key]

        merged_dictionary[key] = new_value

    for key in dict2:
        if key not in merged_dictionary:
            merged_dictionary[key] = dict2[key]

    return merged_dictionary


def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])
