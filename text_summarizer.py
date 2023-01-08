import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import preprocess_data as Pd


def get_sentence(file_name: str):
    lst = Pd.extract_all(file_name)
    cleaned_lst = []
    for item in lst:
        citem = item
        if citem != "" and type(citem) != float:
            cleaned_lst.append(citem)
    return cleaned_lst


text = Pd.one_long_string(get_sentence("SALG-shorten-winter2019.CSV"))
stopWords = set(stopwords.words("english"))
words = word_tokenize(text)


"""
Below is a citation for a portion of the code in this file.
***************************************************************************************
*    Title: Text Summarization in Python
*    Author: Nitin Kumar
*    Date: Nov 18, 2022
*    Code version: Unknown
*    Availability: https://www.mygreatlearning.com/blog/text-summarization-in-python/
*
***************************************************************************************
"""

freqTable = dict()
for word in words:
    word = word.lower()
    if word in stopWords:
        continue
    if word in freqTable:
        freqTable[word] += 1
    else:
        freqTable[word] = 1

sentences = sent_tokenize(text)
sentenceValue = dict()

for sentence in sentences:
    for word, freq in freqTable.items():
        if word in sentence.lower():
            if sentence in sentenceValue:
                sentenceValue[sentence] += freq
            else:
                sentenceValue[sentence] = freq

sumValues = 0
for sentence in sentenceValue:
    sumValues += sentenceValue[sentence]

average = int(sumValues / len(sentenceValue))

summary = ''
for sentence in sentences:
    if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.3 * average)):
        summary += " " + sentence
print(summary)
