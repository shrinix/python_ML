import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from utils import *

# Sample corpus
# documents = ['Machine learning is the study of computer algorithms that improve automatically through experience.\
# Machine learning algorithms build a mathematical model based on sample data, known as training data.\
# The discipline of machine learning employs various approaches to teach computers to accomplish tasks \
# where no fully satisfactory algorithm is available.',
# 'Machine learning is closely related to computational statistics, which focuses on making predictions using computers.\
# The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.',
# 'Machine learning involves computers discovering how they can perform tasks without being explicitly programmed to do so. \
# It involves computers learning from data provided so that they carry out certain tasks.',
# 'Machine learning approaches are traditionally divided into three broad categories, depending on the nature of the "signal"\
# or "feedback" available to the learning system: Supervised, Unsupervised and Reinforcement',
# 'Software engineering is the systematic application of engineering approaches to the development of software.\
# Software engineering is a computing discipline.',
# 'A software engineer creates programs based on logic for the computer to execute. A software engineer has to be more concerned\
# about the correctness of the program in all the cases. Meanwhile, a data scientist is comfortable with uncertainty and variability.\
# Developing a machine learning application is more iterative and explorative process than software engineering.'
# ]

documents = []

pd.set_option('display.max_colwidth', 0)
pd.set_option('display.max_columns', 0)

documents = load_files()

documents_df=pd.DataFrame(documents,columns=['documents'])

# removing special characters and stop words from the text
stop_words_l=stopwords.words('english')
documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )

tfidfvectoriser=TfidfVectorizer(max_features=64)
tfidfvectoriser.fit(documents_df.documents_cleaned)
tfidf_vectors=tfidfvectoriser.transform(documents_df.documents_cleaned)

print(tfidf_vectors.shape)

# Every vector is already normalised to have unit L2 norm
# --> np.linalg.norm(tfidf_vectors[0],ord=2)

tfidf_vectors=tfidf_vectors.toarray()
print (tfidf_vectors[0])

#Every document has been converted into a 64 dimensional vector. As we set the max_features=64

# Pairwise cosine similarity would just be the dot product of the vectors becasue tf-idf vectors from sklearn are already normalized
# and L2 norm of these vectors is 1. So denominator of cosine similiarity formula is 1 in this case

pairwise_similarities=np.dot(tfidf_vectors,tfidf_vectors.T)
pairwise_differences=euclidean_distances(tfidf_vectors)

print (tfidf_vectors[0])
print (pairwise_similarities.shape)

for i in range(0, len(documents)):
    print(pairwise_similarities[i][:])
    print(pairwise_differences[i][:])

#similarity is highest, 1 at index 0 becasue they are the same documents

def most_similar(doc_id,similarity_matrix,matrix):
    print (f'\n Text comparison for: "{documents_df.iloc[doc_id]["documents"]}"') #using {matrix}:
    if matrix=='Cosine Similarity':
        similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix=='Euclidean Distance':
        similar_ix=np.argsort(similarity_matrix[doc_id])
    for ix in similar_ix:
        if ix==doc_id:
            continue
        print (f'{documents_df.iloc[ix]["documents"]} --> {matrix} : {similarity_matrix[doc_id][ix]}')

most_similar(0,pairwise_similarities,'Cosine Similarity')
most_similar(0,pairwise_differences,'Euclidean Distance')
