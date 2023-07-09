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


# use the following code in console to estiment how many clusters we need
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

