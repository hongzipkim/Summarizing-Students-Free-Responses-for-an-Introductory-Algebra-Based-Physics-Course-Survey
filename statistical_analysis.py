"""Statistical analysis for the data given."""
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import statistics

nltk.download('stopwords')
nltk.download('wordnet')
import string

# first read csv in prompt using the code below:
"""
    file = open("name of the file")
    csvreader = csv.reader(file)
"""

# use the following to get the header of the csv file in prompt:
"""
    headers = next(csvreader)
"""


# use the following line to get the responses for each header in headers


# define the function below for cleaning text for a single line of text

def clean_text(text: str) -> list:
    """Cleans the text of the csv file"""
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text.lower() if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc = [word.lower() for word in nopunc.split() if word not in
              stopwords.words('english')]
    return [stemmer.lemmatize(word, 'v') for word in nopunc]


def clean_text_for_nb(text: str) -> str:
    """Cleans the text of the csv file"""
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text.lower() if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc = [word.lower() for word in nopunc.split() if word not in
              stopwords.words('english')]
    listy = [stemmer.lemmatize(word, 'v') for word in nopunc]
    return ' '.join(listy)


# define the function below for cleaning a list of text
cleaned_list = []


def clean_list(listy: list) -> list:
    """
    List is the list of lines that you want to clean.
    this return a nested list, each item in the nested list is the cleaned
    version of the corresponding line in List.
    """
    for item in listy:
        cleaned = clean_text(item)
        cleaned_list.append(cleaned)
    return cleaned_list


def open_file(file: str, title: int) -> list[str]:
    """Opens different columns of the csv file"""
    lst = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if title == 1:
                if row[0] != 'N/A' and row[0] != '' and row[0] != '/' and row[0] != 'n/a' and \
                        row[0] != 'None' and row[0] != 'none' and row[0] != '.':
                    lst.append(row[0])
            elif title == 2:
                if row[1] != 'N/A' and row[1] != '' and row[1] != '/' and row[1] != 'n/a' and \
                        row[1] != 'None' and row[1] != 'none' and row[1] != '.':
                    lst.append(row[1])
            elif title == 3:
                if row[2] != 'N/A' and row[2] != '' and row[2] != '/' and row[2] != 'n/a' and \
                        row[2] != 'None' and row[2] != 'none' and row[2] != '.':
                    lst.append(row[2])
            elif title == 4:
                if row[3] != 'N/A' and row[3] != '' and row[3] != '/' and row[3] != 'n/a' and \
                        row[3] != 'None' and row[3] != 'none' and row[3] != '.':
                    lst.append(row[3])
            elif title == 5:
                if row[4] != 'N/A' and row[4] != '' and row[4] != '/' and row[4] != 'n/a' and \
                        row[4] != 'None' and row[4] != 'none' and row[4] != '.':
                    lst.append(row[4])
            elif title == 6:
                if row[5] != 'N/A' and row[5] != '' and row[5] != '/' and row[5] != 'n/a' and \
                        row[5] != 'None' and row[5] != 'none' and row[5] != '.':
                    lst.append(row[5])
            elif title == 7:
                if row[6] != 'N/A' and row[6] != '' and row[6] != '/' and row[6] != 'n/a' and \
                        row[6] != 'None' and row[6] != 'none' and row[6] != '.':
                    lst.append(row[6])
            elif title == 8:
                if row[7] != 'N/A' and row[7] != '' and row[7] != '/' and row[7] != 'n/a' and \
                        row[7] != 'None' and row[7] != 'none' and row[7] != '.':
                    lst.append(row[7])
            elif title == 9:
                if row[8] != 'N/A' and row[8] != '' and row[8] != '/' and row[8] != 'n/a' and \
                        row[8] != 'None' and row[8] != 'none' and row[8] != '.':
                    lst.append(row[8])
            elif title == 10:
                if row[9] != 'N/A' and row[9] != '' and row[9] != '/' and row[9] != 'n/a' and \
                        row[9] != 'None' and row[9] != 'none' and row[9] != '.':
                    lst.append(row[9])
            elif title == 11:
                if row[10] != 'N/A' and row[10] != '' and row[10] != '/' and row[10] != 'n/a' and \
                        row[10] != 'None' and row[10] != 'none' and row[10] != '.':
                    lst.append(row[10])
            elif title == 12:
                if row[11] != 'N/A' and row[11] != '' and row[11] != '/' and row[11] != 'n/a' and \
                        row[11] != 'None' and row[11] != 'none' and row[11] != '.':
                    lst.append(row[11])
            elif title == 13:
                if row[12] != 'N/A' and row[12] != '' and row[12] != '/' and row[12] != 'n/a' and \
                        row[12] != 'None' and row[12] != 'none' and row[12] != '.':
                    lst.append(row[12])
            elif title == 14:
                if row[13] != 'N/A' and row[13] != '' and row[13] != '/' and row[13] != 'n/a' and \
                        row[13] != 'None' and row[13] != 'none' and row[13] != '.':
                    lst.append(row[13])
            elif title == 15:
                if row[14] != 'N/A' and row[14] != '' and row[14] != '/' and row[14] != 'n/a' and \
                        row[14] != 'None' and row[14] != 'none' and row[14] != '.':
                    lst.append(row[14])
            else:
                raise IndexError
        return lst


def likert_frq_open_file(file: str) -> list[tuple]:
    """Maps average of likert scale to a list of free responses"""
    listy = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            frq = []
            likert = []
            for cell in row:
                if len(cell) == 1:
                    if cell != '-' and cell != '.' and cell != '/' and cell != 'N' and cell != ' '\
                            and cell != 'h' and cell != 'o':
                        likert.append(cell)
                else:
                    frq.append(cell)
            listy.append((likert, frq))
        return listy


def clean_likert_frq_open_file(file: str) -> list[tuple]:
    """Cleans the list of tuples mapping likert to frq"""
    lfrq = likert_frq_open_file(file)
    listy = []
    for item in lfrq:
        for content in item[1]:
            if content != '' and content != 'n/a' and content != 'N/A' and content != '-' \
                and content != 'N' and content != 'Nope' and content != 'nope' and content != 'n' \
                    and content != 'none' and content != 'None' and content != ' ' \
                    and content != 'no' and content != 'No' and content != 'h':
                if item not in listy and '9' not in item[0]:
                    listy.append(item)
                elif item not in listy and file == 'winter2019_improvement.csv':
                    listy.append(item)
                elif item not in listy and file == 'winter2020_improvement.csv':
                    listy.append(item)
                elif item not in listy and file == 'winter2021_improvement.csv':
                    listy.append(item)
    return listy


def median_likert(file: str) -> float:
    """Calculates the median Likert response"""
    lfrq = clean_likert_frq_open_file(file)
    listy = []
    for item in lfrq:
        for likert in item[0]:
            listy.append(int(likert))
    return statistics.median(listy)


def list_of_likert(file: str):
    """Outputs a list of Likert scale response"""
    lfrq = clean_likert_frq_open_file(file)
    likert = []
    for item in lfrq:
        for i in item[0]:
            if int(i) == 1:
                likert.append(1)
            elif int(i) == 2:
                likert.append(2)
            elif int(i) == 3:
                likert.append(3)
            elif int(i) == 4:
                likert.append(4)
            else:
                likert.append(5)
    return likert


def nested_list_of_likert(file: str):
    lfrq = clean_likert_frq_open_file(file)
    likert = []
    for item in lfrq:
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


def untouched_nested_list_of_likert(file: str):
    lfrq = clean_likert_frq_open_file(file)
    likert = []
    for item in lfrq:
        likert.append(item[0])
    return likert


def median_likert_all_years(first: str, second: str, third: str):
    """Median likert of all three years combined"""
    winter2019 = clean_likert_frq_open_file(first)
    winter2020 = clean_likert_frq_open_file(second)
    winter2021 = clean_likert_frq_open_file(third)
    big_list = []
    for item in winter2019:
        for likert in item[0]:
            big_list.append(int(likert))
    for item in winter2020:
        for likert in item[0]:
            big_list.append(int(likert))
    for item in winter2021:
        for likert in item[0]:
            big_list.append(int(likert))
    median = statistics.median(big_list)
    print('The median likert across all three years is: ', median)


def flatten(lister: list[list]) -> list:
    """Makes a nested list only into one list. Only use it after you use the clean_text function"""
    return [x for listy in lister for x in listy]


def one_long_string(listy: list) -> str:
    """Makes the list into one long string"""
    return ' '.join(listy)


if __name__ == "__main__":
    print('Understanding')
    median_likert_all_years('winter2019_understanding.csv', 'winter2020_understanding.csv',
                            'winter2021_understanding.csv')
    print('Skill')
    median_likert_all_years('winter2019_skill.csv', 'winter2020_skill.csv',
                            'winter2021_skill.csv')
    print('Attitude')
    median_likert_all_years('winter2019_attitude.csv', 'winter2020_attitude.csv',
                            'winter2021_attitude.csv')
    print('Integrate')
    median_likert_all_years('winter2019_integrate.csv', 'winter2020_integrate.csv',
                            'winter2021_integrate.csv')
    print('Overall')
    median_likert_all_years('winter2019_overall.csv', 'winter2020_overall.csv',
                            'winter2021_overall.csv')
    print('Activities')
    median_likert_all_years('winter2019_activities.csv', 'winter2020_activities.csv',
                            'winter2021_activities.csv')
    print('Assignments')
    median_likert_all_years('winter2019_assignments.csv', 'winter2020_assignments.csv',
                            'winter2021_assignments.csv')
    print('Resources')
    median_likert_all_years('winter2019_resources.csv', 'winter2020_resources.csv',
                            'winter2021_resources.csv')
    print('Info')
    median_likert_all_years('winter2019_info.csv', 'winter2020_info.csv', 'winter2021_info.csv')
    print('Support')
    median_likert_all_years('winter2019_support.csv', 'winter2020_support.csv',
                            'winter2021_support.csv')
