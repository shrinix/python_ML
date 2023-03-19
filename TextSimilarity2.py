import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
from sklearn.feature_extraction.text import CountVectorizer

from utils import *

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

## Create a list of the sentences
texts, indexes, short_col_names = load_files()

## Firstly let's count the words using the CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()
matrix = count_vectorizer.fit_transform(texts)

## we can create a dataframe to represent the number of the words in every sentence
table = matrix.todense()
df = pd.DataFrame(table,
                  columns=count_vectorizer.get_feature_names_out(),
                  index=indexes)

print('-----Dataframe-----')
print(df)

## Compute the Euclidean distance of these sentences
from scipy.spatial import distance

matrix = distance.cdist(df, df, 'euclidean')

df_eucl = pd.DataFrame(matrix, columns=texts, index=indexes)

print('-----Euclidean Distance-----')
print(df_eucl)

## Constract again the bag of words table

count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()
matrix = count_vectorizer.fit_transform(texts)

## Creating a data frame to represent the number of the words in every sentence
table = matrix.todense()
df = pd.DataFrame(table,columns=count_vectorizer.get_feature_names_out(), index=texts)

## Aplying the Cosine similarity module
from sklearn.metrics.pairwise import cosine_similarity

values = cosine_similarity(df, df)
df = pd.DataFrame(values, columns=texts, index=indexes)

print('-----Cosine Similarity-----')
print(df)

#sns.heatmap(df, annot=True, fmt="g", cmap='viridis')
#plt.show()
