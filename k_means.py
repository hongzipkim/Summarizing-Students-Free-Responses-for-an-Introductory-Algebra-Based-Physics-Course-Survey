from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
nltk.download('omw-1.4')
nltk.download('punkt')
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

import csv
from typing import List, Type
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import re
nltk.download('stopwords')
nltk.download('wordnet')
import string

###############################################################################

# define the following function to count the number of unique word in a text


def unique_word(file_name: str):
    """
    Parameters
    ----------
    file_name: str
        This is the file that contains the students' responses you want to
        analyze.
    title: int
        This is the column that contains the students' responses you want to
        analyze.

    Returns
    -------
    This function returns the number of unique words in your input.

    """
    lst = extract_all(file_name)
    text = one_long_string(lst)
    cleaned_text = clean_text(text)
    return len(sorted(set(cleaned_text)))


# use the following code in the console to estimate how many clusters we need
# THE ELBOW METHOD

def elbow(file_name: str, endpoint: int) -> plt:
    """
    Parameters
    ----------
    file_name: str
        This is the file that contains the students' responses you want to
        analyze.

    variable_number: int
        This cannot exceed the number of data points that you have in your
        dataset.
        You can get the number of data point in your file using unique_word.
        Don't put a number larger than 20 though, it takes forever to run.

    Returns
    -------
    This function returns a elbow plot figure showcasing what might be the
    optimum number of cluster that we want to select for kmeans.

    """
    Sum_of_squared_distances = []

    K = range(1, endpoint)

    ext = extract_all(file_name)

    s = one_long_string(ext)

    c = clean_text(s)

    l = one_long_string(c)

    vectorized_text = vectorization(l)

    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(vectorized_text)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


# use the following code to perform kmeans clustering
def cluster(file_name: str , num_of_cluster: int):
    """
    K means clustering :D
    file_name is the name of the file that you want to analyze.
    num_of_cluster is the number of clusters you want your plot to have.
    """
    df = pdframe(file_name)
    cdf = clean_df(df)
    tdf = tfidf_df(cdf)

    # initialize kmeans with num_of_cluster centroids
    kmeans = KMeans(n_clusters=num_of_cluster, random_state=50)
    # fit the model
    kmeans.fit(tdf)
    # store cluster labels in a variable
    clusters = kmeans.labels_

    df["clusters"] = clusters
    return clusters

# using tsne to lower the dimensions
# this lower the dimensions of datapoint using t_sne
def lower_d(file_name: str):
    df = pdframe(file_name)
    cdf = clean_df(df)
    tdf = tfidf_df(cdf)

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=200)
    tsne_obj = tsne.fit_transform(tdf)
    # tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
    #                     'Y':tsne_obj[:,1]})
    x0 = tsne_obj[:, 0]
    x1 = tsne_obj[:, 1]
    df['x0'] = x0
    df['x1'] = x1
    return x0,x1,df

# this lower the dimensions of datapoint using pca
def lower_d_pca(file_name: str):
    df = pdframe(file_name)
    cdf = clean_df(df)
    tdf = tfidf_df(cdf)
    pca = PCA(n_components=2, random_state = 42)
    # pass our tdf to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(tdf.toarray())
    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]
    df['x0'] = x0
    df['x1'] = x1
    return x0, x1, pca_vecs
  

def keywords_in_cluster(file_name: str, clusters, n_terms: int):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pdframe(file_name)
    cdf = clean_df(df)
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5)
    tdf = vectorizer.fit_transform(cdf["cleaned"])
    ddf = pd.DataFrame(tdf.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names_out() # access tf-idf terms
    for i,r in ddf.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) 


# use the method below to find the frequent words in the whole dataset
def keywords(file_name: str, n_term: int):
    lst = extract_all(file_name)
    text = one_long_string(lst)
    ltext = text.lower()
    cleaned_text = clean_text(ltext)
    counting = Counter(cleaned_text)
    keyword = counting.most_common(n_term)
    return keyword

##########################################################################################

# here is the code for generating cluster results, I use 2019 data as an example here
df = pdframe("SALG-shorten-winter2019.CSV")
cdf = clean_df(df)
tdf = tfidf_df(cdf)
pca = PCA(n_components=10)
X_pca = pca.fit_transform(tdf.toarray()) 
tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=5000, learning_rate=100)
tsne_results = tsne.fit_transform(X_pca)
df["x0"] = tsne_results[:, 0]
df["x1"] = tsne_results[:, 1]

# initialize kmeans with num_of_cluster centroids
kmeans = KMeans(n_clusters=12, random_state=50)
# fit the model
kmeans.fit(tsne_results)

label=kmeans.predict(tsne_results)
print('Silhouette Score(n=12):', silhouette_score(tsne_results,label)) 
##########################################################################################

# using elbow method, create a range of number of clusters to test
K = range(1,15)

# run k-means clustering for each k and save the inertia (sum of squared distances)
inertias = []
for k in K:
    model = KMeans(n_clusters=k, random_state=50)
    model.fit(tsne_results)
    inertias.append(model.inertia_)

# plot the elbow curve to find the optimal number of clusters, I use 2019 data as an example here, no significant elbow point was found.
plt.plot(K, inertias, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for 2019 SALG Data')
plt.show()

##########################################################################################

# Use the following to get sentences in clusters
def get_sentence(file_name: str):
  lst = extract_all(file_name)
  cleaned_lst = []
  for item in lst:
    citem = item
    if citem != "" and citem.isdigit() == False:
      cleaned_lst.append(citem)
  return cleaned_lst

def nice_pf(file_name: str, num_of_cluster: int):
    """ This produces a data frame with the original student responses,
    the cleaned students' responses, the cluster, and the x0 and x1 component
    of the vector.
    """
    df = pdframe(file_name)
    cdf = clean_df(df) 
    tdf = tfidf_df(cdf)
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(tdf.toarray()) 
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=5000, learning_rate=100)
    tsne_results = tsne.fit_transform(X_pca)
    df["x0"] = tsne_results[:, 0]
    df["x1"] = tsne_results[:, 1]

    # initialize kmeans with num_of_cluster centroids
    kmeans = KMeans(n_clusters=num_of_cluster, random_state=50)
    # fit the model
    kmeans.fit(tsne_results)

    label=kmeans.predict(tsne_results)

    cdf["clusters"] = label
    return cdf

def find_sentence(good_pf, location: int, word: str):
  df = good_pf.loc[good_pf["clusters"] == location]
  sentence = []
  for row in df['Responses']:
    if word in row:
      sentence.append(row)
  return sentence

def get_sentence_in_cluster(good_pf, location):
  df = good_pf.loc[good_pf["clusters"] == location]
  sentence = []
  for row in df['Responses']:
    sentence.append(row)
  return sentence

##########################################################################################
# This get sentences in clusters
data = pdframe("SALG-shorten-winter2019.CSV")
pf = nice_pf("SALG-shorten-winter2019.CSV", 12)
i = 0
while i < 12:
  sec = get_sentence_in_cluster(pf, i)
  print(len(sec))
  print(sec)
  print("______________________________")
  i = i + 1

# This gets the keywords in clusters
keywords_in_cluster("SALG-shorten-winter2019.CSV", label, 30) # 30 is the number of keywords that we got for our paper, you can replace it with another number
##########################################################################################
# This is an example of plotting the clustering result
df["Cluster"] = label

cluster_map = {0: "Suggestions", 1: "Workload", 2: "Understanding of Concepts", 
               3:"General Comments", 4:"Comments on the Subject", 5:"Helpfulness of Different Resources",
               6:"Problem Solving", 7:"Learning Gains", 8:"Lecture & Textbook", 9:"Learning Methods", 
               10:"Graded Assignments & Test", 11:"Participation"}

df['Cluster'] = df['Cluster'].map(cluster_map)

plt.figure(figsize=(10, 5))
# set a title
plt.title("K-means Clustering of PHY132 Students' Free Responses in Winter 2019 SALG Survey", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("comp1 of t-SNE", fontdict={"fontsize": 16})
plt.ylabel("comp2 of t-SNE", fontdict={"fontsize": 16})
# create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=df, x="x0", y='x1', hue='Cluster', palette = "Paired")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()

################################################################################
# Now we generate a summary of a certain cluster that you want to analyze for a particular year
pf2019 = nice_pf("SALG-shorten-winter2019.CSV", 12)
# The input text now becomes:
text = one_long_string(get_sentence_in_cluster(pf2019, 4)), # 4 signifies we want to generate a summary for the fourth clusters in the cluster result of 2019 data
stopWords = set(stopwords.words("english"))
words = word_tokenize(text)

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
count = 0
for sentence in sentences:
    if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.3 * average)):
      if count <= 5 and count != 0:
        if sentence[-1:] == '.':
          summary = summary + " " + sentence
        elif sentence[-1:] != '.':
          summary = summary + " " + sentence + ". "
      elif count <= 5 and count == 0:
        if sentence[-1:] == '.':
          summary = sentence
        elif sentence[-1:] != '.':
          summary = sentence + ". "
      count += 1
print(summary)
################################################################################
# In addition, we can use the 'find_sentence' function to find sentences containing a particular keyword in a given cluster
# Here is an example
pf2021 = nice_pf("SALG-shorten-winter2021.CSV", 10)
find_sentence(pf2021, 3, "study") # where 3 signifies the third cluster and 'study' is the keyword
