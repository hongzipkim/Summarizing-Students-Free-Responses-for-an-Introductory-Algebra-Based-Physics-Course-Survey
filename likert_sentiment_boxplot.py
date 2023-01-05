import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statistical_analysis as sta
import sentiment_analysis as senta
import seaborn as sns
from matplotlib.patches import PathPatch

def get_the_total_data(title: str, first: str, second: str, third: str):
    winter2019_likert = sta.nested_list_of_likert(first)
    winter2020_likert = sta.nested_list_of_likert(second)
    winter2021_likert = sta.nested_list_of_likert(third)
    winter2019_sentiment = senta.individual_sentiment(first)
    winter2020_sentiment = senta.individual_sentiment(second)
    winter2021_sentiment = senta.individual_sentiment(third)
    likert = []
    sentiment = []
    group = []
    for sent in winter2019_sentiment:
        for each_sent in sent:
            for like in winter2019_likert:
                for each in like:
                    likert.append(each)
                    group.append(title)
                    sentiment.append(each_sent)
    for sent in winter2020_sentiment:
        for each_sent in sent:
            for like in winter2020_likert:
                for each in like:
                    likert.append(each)
                    group.append(title)
                    sentiment.append(each_sent)
    for sent in winter2021_sentiment:
        for each_sent in sent:
            for like in winter2021_likert:
                for each in like:
                    likert.append(each)
                    group.append(title)
                    sentiment.append(each_sent)
    total_data = {'group': group, 'likert': likert, 'sentiment': sentiment}

    return total_data


def new_boxplot():
    topic1 = get_the_total_data('understanding', 'winter2019_understanding.csv', 'winter2020_understanding.csv',
                                'winter2021_understanding.csv')
    topic2 = get_the_total_data('skill', 'winter2019_skill.csv', 'winter2020_skill.csv',
                                'winter2021_skill.csv')
    topic3 = get_the_total_data('attitude', 'winter2019_attitude.csv', 'winter2020_attitude.csv',
                                'winter2021_attitude.csv')
    topic4 = get_the_total_data('integrate', 'winter2019_integrate.csv', 'winter2020_integrate.csv',
                                'winter2021_integrate.csv')
    topic5 = get_the_total_data('overall', 'winter2019_overall.csv', 'winter2020_overall.csv',
                                'winter2021_overall.csv')
    topic6 = get_the_total_data('activities', 'winter2019_activities.csv', 'winter2020_activities.csv',
                                'winter2021_activities.csv')
    topic7 = get_the_total_data('assignments', 'winter2019_assignments.csv', 'winter2020_assignments.csv',
                                'winter2021_assignments.csv')
    topic8 = get_the_total_data('resources', 'winter2019_resources.csv', 'winter2020_resources.csv',
                                'winter2021_resources.csv')
    topic9 = get_the_total_data('info', 'winter2019_info.csv', 'winter2020_info.csv',
                                        'winter2021_info.csv')
    topic10 = get_the_total_data('support', 'winter2019_support.csv', 'winter2020_support.csv',
                                 'winter2021_support.csv')
    d0 = merge_dictionaries(topic1, topic2)
    d1 = merge_dictionaries(d0, topic3)
    d2 = merge_dictionaries(d1, topic4)
    d3 = merge_dictionaries(d2, topic5)
    d4 = merge_dictionaries(d3, topic6)
    d5 = merge_dictionaries(d4, topic7)
    d6 = merge_dictionaries(d5, topic8)
    d7 = merge_dictionaries(d6, topic9)
    d8 = merge_dictionaries(d7, topic10)
    df = pd.DataFrame(data=d8)
    print(df)
    dd = pd.melt(df, id_vars=['group'], value_vars=['sentiment', 'likert'], var_name='type')
    fig = plt.figure(figsize=(20, 12))
    rgb = [(255, 204, 153),
           (128, 229, 255)]
    colors = [tuple(t/255 for t in x) for x in rgb]
    b = sns.boxplot(x='group', y='value', data=dd, hue='type', medianprops=dict(color="red", alpha=0.9), palette=colors)
    group = ['understanding', 'skill', 'attitude', 'integrate', 'overall', 'activities', 'assignments', 'resources',
             'info', 'support']
    b.set_yticklabels(b.get_yticks(), size=15)
    b.set_xticklabels(group, size=15)
    b.set_xlabel("section", fontsize=20)
    b.set_ylabel("score", fontsize=20)
    adjust_box_widths(fig, 0.9)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=13)

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


def test(title: str, first: str):
    winter2019_likert = sta.nested_list_of_likert(first)
    winter2019_sentiment = senta.individual_sentiment(first)
    likert = []
    sentiment = []
    group = []
    for sent in winter2019_sentiment:
        for each_sent in sent:
            for like in winter2019_likert:
                for each in like:
                    likert.append(each)
                    group.append(title)
                    sentiment.append(each_sent)
    total_data = {'group': group, 'likert': likert, 'sentiment': sentiment}
    df = pd.DataFrame(data=total_data)
    print(df)
