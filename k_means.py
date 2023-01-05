
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

import preprocess_data as Pd


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
    lst = Pd.extract_all(file_name)
    text = Pd.one_long_string(lst)
    cleaned_text = Pd.clean_text(text)
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

    ext = Pd.extract_all(file_name)

    s = Pd.one_long_string(ext)

    c = Pd.clean_text(s)

    l = Pd.one_long_string(c)

    vectorized_text = Pd.vectorization(l)

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
def cluster(file_name: str , num_of_cluster: int) -> plt:
    """
    K means clustering :D
    file_name is the name of the file that you want to analyze.
    num_of_cluster is the number of clusters you want your plot to have.

    The output is a plot.
    """
    df = Pd.pdframe(file_name)
    cdf = Pd.clean_df(df)
    tdf = Pd.tfidf_df(cdf)

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
    df = Pd.pdframe(file_name)
    cdf = Pd.clean_df(df)
    tdf = Pd.tfidf_df(cdf)

    tsne = TSNE()
    tsne_obj = tsne.fit_transform(tdf)

    x0 = tsne_obj[:, 0]
    x1 = tsne_obj[:, 1]
    df['x0'] = x0
    df['x1'] = x1
    return x0, x1, tsne_obj

# this lower the dimensions of datapoint using pca
def lower_d_pca(file_name: str):
    df = Pd.pdframe(file_name)
    cdf = Pd.clean_df(df)
    tdf = Pd.tfidf_df(cdf)
    pca = PCA(n_components=2, random_state=42)
    # pass our X to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(tdf.toarray())
    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]
    df['x0'] = x0
    df['x1'] = x1
    return x0, x1, pca_vecs

def lower_d_kpca(file_name: str):
    df = Pd.pdframe(file_name)
    cdf = Pd.clean_df(df)
    tdf = Pd.tfidf_df(cdf)
    kpca = KernelPCA(kernel='rbf',fit_inverse_transform=True, gamma=2)
    kpca_vecs = kpca.fit_transform(tdf.toarray())

    x0 = kpca_vecs[:, 0]
    x1 = kpca_vecs[:, 1]
    df['x0'] = x0
    df['x1'] = x1
    return x0, x1, kpca_vecs

def keywords_in_cluster(file_name: str, clusters, n_terms: int):
    """This function returns the keywords for each centroid of the KMeans"""
    df = Pd.pdframe(file_name)
    cdf = Pd.clean_df(df)
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
    tdf = vectorizer.fit_transform(cdf["cleaned"])
    ddf = pd.DataFrame(tdf.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names() # access tf-idf terms
    for i,r in ddf.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) # for each row of the dataframe, find the n terms that have the highest tf idf score
"""
cluster_map_2019 = {0: "General Comments", 1: "Learning Gains", 2: "Class Activities", 3:"Homework", 4:"Participation", 5:"Learning Resources", 6:"Graded Assignments & Test", 7:"Practical", 8:"Applying Skills", 9:"Suggestions", 10:"Use of Piazza", 11:"Understanding of Class Content"}
cluster_map_2020 = {0:"Course Structure", 1:"Supports", 2:"Learning Resources", 3:"Participation", 4:"Problem Solving", 5:"Study Methods", 6:"Applying Skills", 7:"Understanding of Class Content", 8:"Practical", 9:"Learning Gain", 10:"Online Class"}
cluster_map_2021 = {0:"Graded Assignment & Test", 1:"Participation and Lecture Location", 2:"Understanding of Class Content", 3:"Practical", 4:"Learning Methods", 5:"General Comments", 6:"Attitude", 7:"Applying Skills", 8:"Learning Resources", 9:"Suggestions"}


"""

# use the following code to display results from kmeans clustering
# map clusters to appropriate labels
"""
df = Pd.pdframe("SALG-shorten-winter2019.CSV")
cdf = Pd.clean_df(df)
tdf = Pd.tfidf_df(cdf)
pca = PCA(n_components=10)
X_pca = pca.fit_transform(tdf.toarray()) 
tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=5000, learning_rate=100)
tsne_results = tsne.fit_transform(X_pca)
df["x0"] = tsne_results[:, 0]
df["x1"] = tsne_results[:, 1]

clustering = Pd.cluster("SALG-shorten-winter2019.CSV", 12)
df["Cluster"] = clustering
cluster_map = {0: "General Comments", 1: "Learning Gains", 2: "Class Activities", 3:"Homework", 4:"Participation", 5:"Learning Resources", 6:"Graded Assignments & Test", 7:"Practical", 8:"Applying Skills", 9:"Suggestions", 10:"Use of Piazza", 11:"Understanding of Class Content"}
# apply mapping
df['Cluster'] = df['Cluster'].map(cluster_map)

plt.figure(figsize=(10, 5))
# set a title
plt.title("K-means Clustering of PHY132 Students' Free Responses in Winter 2019 SALG Survey", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("comp1", fontdict={"fontsize": 16})
plt.ylabel("comp2", fontdict={"fontsize": 16})
# create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=df, x="x0", y='x1', hue='Cluster', palette = "Paired")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()
"""

# use the method below to find the frequent words in the whole dataset
def keywords(file_name: str, n_term: int):
    lst = Pd.extract_all(file_name)
    flatten_list = Pd.flatten(lst)
    text = Pd.one_long_string(flatten_list)
    cleaned_text = Pd.clean_text(text)
    counting = Counter(cleaned_text)
    keyword = counting.most_common(n_term)
    return keyword

# use the method below to create a barplot for the most frequent words
def keywords_plot(keywords: list[tuple]):
    df = pd.DataFrame(keywords, columns =["Word", "Count"])
    #mp = df.plot(x="Word", y="Count", kind="bar", figsize=(10, 9), color = "")

# displaying bar graph
    #mp.show()
    p = df.plot.barh(x='Word', y='Count',
                     title="Counts for the Top 10 Most Frequent Words",
                     figsize=(10, 9));

    p.show(block=True)


#Importing Libraries

"""
from wordcloud import WordCloud,STOPWORDS

line = Pd.extract_all("SALG-shorten-winter2021.CSV")
cl = Pd.one_long_string(line)
cl = Pd.cl.lower()
ct = Pd.clean_text(cl)
cs = Pd.one_long_string(ct)


wordcloud = WordCloud(width = 800, height = 800,
                background_color = "white", max_words = 50,
                min_font_size = 29).generate(cs)
 
# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()
"""

def nice_pf(file_name: str, num_of_cluster: int):
    """ This produce a data frame with the original student responses,
    the cleaned students' responses, the cluster, and the x0 and x1 component
    of the vector.
    """
    df = Pd.pdframe(file_name)
    cdf = Pd.clean_df(df)
    clusters = cluster(file_name, num_of_cluster)
    cdf["clusters"] = clusters
    return cdf

def find_sentence(good_pf, location: int, word: str):
    df = good_pf.loc[good_pf["clusters"] == location]
    sentence = []
    for row in df['Responses']:
        if word in row:
            sentence.append(row)
    return sentence

"""
data = pdframe("SALG-shorten-winter2021.CSV")
pf = nice_pf("SALG-shorten-winter2021.CSV", 10)
find_sentence(pf, 0, "course")
"""


"""
clustering = cluster("SALG-shorten-winter2019.CSV", 12)
clustering = cluster("SALG-shorten-winter2020.CSV", 11)
k = cluster("SALG-shorten-winter2021.CSV", 10)

"""
