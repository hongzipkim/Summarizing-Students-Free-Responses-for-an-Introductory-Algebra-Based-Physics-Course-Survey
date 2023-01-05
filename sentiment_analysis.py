"""Code for VADER sentiment analysis"""
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import statistical_analysis as sa
import statistics
import pandas as pd


def sorting_sentiment_score(first: str, second: str, third: str):
    sid_obj = SentimentIntensityAnalyzer()
    listy = []
    analyze2019 = sa.clean_likert_frq_open_file(first)
    analyze2020 = sa.clean_likert_frq_open_file(second)
    analyze2021 = sa.clean_likert_frq_open_file(third)
    for item in analyze2019:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)
            listy.append((sentence, str(sentiment_dict['compound'])))
    for item in analyze2020:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)
            listy.append((sentence, str(sentiment_dict['compound'])))
    for item in analyze2021:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)
            listy.append((sentence, str(sentiment_dict['compound'])))
    new_listy = sorted(listy, key=lambda x: float(x[1]), reverse=True)
    df = pd.DataFrame(new_listy, columns=['Sentence', 'Sentiment Score'])
    return df


def text_sentiment_score(text: str):
    """Analyzes the sentiment of each sentence"""
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(text)
    return sentiment_dict['compound']


def sentiment_scores(file: str):
    """Analyzes the sentiment of each sentence"""
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    analyze = sa.clean_likert_frq_open_file(file)
    for item in analyze:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)
            print("The sentence is: ", sentence)
            print("Overall sentiment dictionary is : ", sentiment_dict)
            print("sentence was rated as ", sentiment_dict['neg'] * 100, "% Negative")
            print("sentence was rated as ", sentiment_dict['neu'] * 100, "% Neutral")
            print("sentence was rated as ", sentiment_dict['pos'] * 100, "% Positive")

            print("Sentence Overall Rated As", end=" ")

            # decide sentiment as positive, negative and neutral
            if sentiment_dict['compound'] >= 0.05:
                print("Positive")

            elif sentiment_dict['compound'] <= - 0.05:
                print("Negative")

            else:
                print("Neutral")


def individual_sentiment(file: str):
    """Outputs a list of the individual sentiment of each sentence"""
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_list = []
    analyze = sa.clean_likert_frq_open_file(file)
    for item in analyze:
        listy = []
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)
            listy.append(sentiment_dict['compound'])
        sentiment_list.append(listy)
    return sentiment_list


def individual_sentiment_for_checking(file: str):
    """Outputs a list of the individual sentiment of each sentence"""
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_list = []
    analyze = sa.clean_likert_frq_open_file(file)
    for item in analyze:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)
            sentiment_list.append(sentiment_dict['compound'])
    return sentiment_list


def average_sentiment(file: str):
    """Returns the average of the sentiments given a data file"""
    sid_obj = SentimentIntensityAnalyzer()

    analyze = sa.clean_likert_frq_open_file(file)
    count = 0
    sum = 0
    for item in analyze:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)
            sum += sentiment_dict['compound']
            count += 1
    average = sum / count
    print("The average sentiment is: ", average)


def just_a_list(file: str):
    analyze = sa.clean_likert_frq_open_file(file)
    listy = []
    for item in analyze:
        for sentence in item[1]:
            listy.append(sentence)
    return listy


def average_sentiment_all_years(first: str, second: str, third: str):
    """Returns the average sentiment of all years"""
    sid_obj = SentimentIntensityAnalyzer()

    analyze_2019 = sa.clean_likert_frq_open_file(first)
    analyze_2020 = sa.clean_likert_frq_open_file(second)
    analyze_2021 = sa.clean_likert_frq_open_file(third)

    count = 0
    sum = 0

    for item in analyze_2019:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)
            sum += sentiment_dict['compound']
            count += 1

    for item in analyze_2020:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)
            sum += sentiment_dict['compound']
            count += 1

    for item in analyze_2021:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)
            sum += sentiment_dict['compound']
            count += 1

    average = sum / count
    print("The average across all three years is: ", average)


def median_sentiment(file: str):
    """Returns the average of the sentiments given a data file"""
    sid_obj = SentimentIntensityAnalyzer()

    analyze = sa.clean_likert_frq_open_file(file)
    median = []
    for item in analyze:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)
            median.append(sentiment_dict['compound'])
    median_score = statistics.median(median)
    print("The average sentiment is: ", median_score)


def median_sentiment_all_years(first: str, second: str, third: str):
    """Returns the average sentiment of all years"""
    sid_obj = SentimentIntensityAnalyzer()

    analyze_2019 = sa.clean_likert_frq_open_file(first)
    analyze_2020 = sa.clean_likert_frq_open_file(second)
    analyze_2021 = sa.clean_likert_frq_open_file(third)

    median = []

    for item in analyze_2019:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)
            median.append(sentiment_dict['compound'])

    for item in analyze_2020:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)
            median.append(sentiment_dict['compound'])

    for item in analyze_2021:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)
            median.append(sentiment_dict['compound'])

    median_score = statistics.median(median)
    print("The average across all three years is: ", median_score)


def count_sentiment(file: str):
    """Counts the number of positive, negative, neutral responses given a data file"""
    sid_obj = SentimentIntensityAnalyzer()
    analyze = sa.clean_likert_frq_open_file(file)
    pos_count = 0
    neg_count = 0
    neu_count = 0

    for item in analyze:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)

            if sentiment_dict['compound'] >= 0.05:
                pos_count += 1

            elif sentiment_dict['compound'] <= - 0.05:
                neg_count += 1
            else:
                neu_count += 1

    print("The total number of positive response is: ", pos_count)
    print("The total number of negative response is: ", neg_count)
    print("The total number of neutral response is: ", neu_count)


def count_sentiment_for_f1(file: str) -> list:
    """
    Counts the number of positive, negative, neutral responses given a data file using it for
    kappa coefficient
    """
    sid_obj = SentimentIntensityAnalyzer()
    listy = []
    analyze = sa.likert_frq_open_file(file)
    for item in analyze:
        for sentence in item[1]:
            sentiment_dict = sid_obj.polarity_scores(sentence)

            if sentiment_dict['compound'] >= 0.05:
                listy.append('Positive')

            elif sentiment_dict['compound'] <= - 0.05:
                listy.append('Negative')
            else:
                listy.append('Neutral')
    return listy


def catch_data(first: str, second: str, third: str):
    """Gets the data needed to describe all data"""
    winter2019_sentiment = individual_sentiment_for_checking(first)
    winter2020_sentiment = individual_sentiment_for_checking(second)
    winter2021_sentiment = individual_sentiment_for_checking(third)
    winter2019_likert = sa.list_of_likert(first)
    winter2020_likert = sa.list_of_likert(second)
    winter2021_likert = sa.list_of_likert(third)
    sentiment = []
    likert = []
    for sent in winter2019_sentiment:
        sentiment.append(sent)
    for sent in winter2020_sentiment:
        sentiment.append(sent)
    for sent in winter2021_sentiment:
        sentiment.append(sent)
    for like in winter2019_likert:
        likert.append(like)
    for like in winter2020_likert:
        likert.append(like)
    for like in winter2021_likert:
        likert.append(like)
    sentiment_data = {'sentiment': sentiment}
    likert_data = {'likert': likert}
    df1 = pd.DataFrame(sentiment_data)
    df2 = pd.DataFrame(likert_data)
    frames = [df1, df2]
    result = pd.concat(frames)
    return result.describe().round(3)


if __name__ == "__main__":
    print('Understanding')
    understanding = catch_data('winter2019_understanding.csv', 'winter2020_understanding.csv',
                                'winter2021_understanding.csv')
    print('Skill')
    skill = catch_data('winter2019_skill.csv', 'winter2020_skill.csv',
                                'winter2021_skill.csv')
    print('Attitude')
    attitude = catch_data('winter2019_attitude.csv', 'winter2020_attitude.csv',
                                'winter2021_attitude.csv')
    print('Integrate')
    integrate = catch_data('winter2019_integrate.csv', 'winter2020_integrate.csv',
                                'winter2021_integrate.csv')
    print('Overall')
    overall = catch_data('winter2019_overall.csv', 'winter2020_overall.csv',
                                'winter2021_overall.csv')
    print('Activities')
    activities = catch_data('winter2019_activities.csv', 'winter2020_activities.csv',
                                'winter2021_activities.csv')
    print('Assignments')
    assignments = catch_data('winter2019_assignments.csv', 'winter2020_assignments.csv',
                                'winter2021_assignments.csv')
    print('Resources')
    resources = catch_data('winter2019_resources.csv', 'winter2020_resources.csv',
                                'winter2021_resources.csv')
    print('Info')
    info = catch_data('winter2019_info.csv', 'winter2020_info.csv', 'winter2021_info.csv')

    print('Support')
    support = catch_data('winter2019_support.csv', 'winter2020_support.csv',
                                'winter2021_support.csv')
    print('Improvement')
    improvement = catch_data('winter2019_improvement.csv', 'winter2020_improvement.csv',
                                'winter2021_improvement.csv')
