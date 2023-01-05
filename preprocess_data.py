"""Code to pre-process the data file given."""
import csv
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
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
        return lst[1:]


def different_open_file(file: str, title: int) -> list[str]:
    """Opens different columns of the csv file"""
    lst = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for cell in row:
                if title == 1:
                    if cell == '':
                        return lst
                    else:
                        lst.append(cell)
    return lst


# use the following code to extract all columns in a given csv file
def extract_all(file: str):
    lst = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for rows in reader:
            for item in rows:
                if item != 'N/A' and item != '' and item != '/' and \
                    item != 'n/a' and item != 'None' and \
                        item != 'none' and item != '.' and \
                            item != '-' and item != 'N' and item != 'Nope' \
                            and item != 'nope' and item != ' ' and item != 'No' \
                                and item != 'no' and item not in "0101121314156789":
                    lst.append(item)
                else:
                    pass

    return lst # this lst is not cleaned, remember to clean it



def flatten(lister: list[list]) -> list:
    """Makes a nested list only into one list. Only use it after you use the clean_text function"""
    return [x for listy in lister for x in listy]


def one_long_string(listy: list) -> str:
    """Makes the list into one long string"""
    return ' '.join(listy)


# This code is for vectorize text that is not yet cleaned
# Use this code after using one_long_string on the extracted column of responses
def vectorization(text):
    """
    Vertorize the input text. In this research, the input could be students'
    responses from a column.
    The detailed step for vectorizing a column of students' responses in the
    dataset will be:
        1. string = one_long_string(open_file("wanted file",wanted column))
        2. vectorization(string)

    """
    x_text = clean_text(text)
    tfidfconvert = TfidfVectorizer(analyzer=clean_text).fit(x_text)
    x_vect = tfidfconvert.transform(x_text)
    return x_vect

###############################################################################

def pdframe(file_name: str):
    fd = extract_all(file_name)
    df = pd.DataFrame(fd, columns=["Responses"])

    return df


def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """
    This function is for cleaning student responses, removing stopwords, links, number and special characters.
    """

    text = re.sub(r"http\S+", "", text)
    text = re.sub("[^A-Za-z]+", " ", text)
    if remove_stopwords:
        tokens = nltk.word_tokenize(text)
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        text = " ".join(tokens)
    text = text.lower().strip()
    return text


def clean_df(df):
    """
    The input is a pandas dataframe.

    This function will clean the dataframe.
    """
    df['cleaned'] = df['Responses'].apply(
        lambda x: preprocess_text(x, remove_stopwords=True))
    return df


def tfidf_df(df):
    """
    Getting the tf-idf of the input data frame.
    
    The output is a data frame as well.
    """
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
    X = vectorizer.fit_transform(df['cleaned'])
    return X
