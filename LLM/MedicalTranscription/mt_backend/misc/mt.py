import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
seed(1)
from matplotlib import style
style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

# Tokenizing 
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize

def read_data(path_and_file):
    df = pd.read_csv('data.csv')
    return df

def get_transcription_data(data):

    data['transcription']=data['transcription'].astype('str')
    data['transcription'] = data['transcription'].str.lower()

    #getting rid of targeted charachters in the trascription
    chars = ['#',':,',': ,',';','$','!','?','*','``','1. ', '2. ', '3. ', '4. ', '5. ','6. ','7. ','8. ','9. ','10. ']
    for c in chars:
        data['transcription'] = data['transcription'].str.replace(c,"")

    #getting rid of targeted charachters in the trascription
    chars = [",", ".", "[", "]", ":", "``", ")", "(", "1", "2", "5", "%", "3", "4", "4-0", "3-0", "6", "''", "0", "2-0", "8", "7", "&", "5-0", "9", "0.5", "1.5", "500", "50", "100", "6-0", "15", "2.5", "14-15", "60", "'", "300", "14", "________", "7-0", "90", "__________", "3.5", "1:100,000", "70", "0.", "80", "1:50,000", "03/08/200 ", "03/09/2007", "25605", "7.314", "33.0", "855.", "08/22/03", "10/500", "125.", "144/6"]
    for c in chars:
        data['transcription'] = data['transcription'].str.replace(c," ")

    return data['transcription']

#Incomplete function
def clean_and_preprocess_data(data):
    data= data.dropna(axis = 0, how ='any') 
    data['transcription']=data['transcription'].astype('str')
    data['transcription'] = data['transcription'].str.lower()

    #getting rid of targeted charachters in the trascription
    chars = ['#',':,',': ,',';','$','!','?','*','``','1. ', '2. ', '3. ', '4. ', '5. ','6. ','7. ','8. ','9. ','10. ']
    for c in chars:
        data['transcription'] = data['transcription'].str.replace(c,"")

    #getting rid of targeted charachters in the trascription
    chars = [",", ".", "[", "]", ":", "``", ")", "(", "1", "2", "5", "%", "3", "4", "4-0", "3-0", "6", "''", "0", "2-0", "8", "7", "&", "5-0", "9", "0.5", "1.5", "500", "50", "100", "6-0", "15", "2.5", "14-15", "60", "'", "300", "14", "________", "7-0", "90", "__________", "3.5", "1:100,000", "70", "0.", "80", "1:50,000", "03/08/200 ", "03/09/2007", "25605", "7.314", "33.0", "855.", "08/22/03", "10/500", "125.", "144/6"]
    for c in chars:
        data['transcription'] = data['transcription'].str.replace(c," ")

    data['tokenized_sents'] = data['transcription'].apply(nltk.word_tokenize)
    print(nltk.tag.pos_tag(data["tokenized_sents"][0]))
    data['POSTags'] = data['tokenized_sents'].apply(pos_tag)

    print(nltk.tag.pos_tag(data["POSTags"][0]))

    # Selecting the nouns in our corpus
    data['Nouns'] = data['POSTags'].apply(lambda x: [(t[0], t[1]) for t in x if t[1]=='NN' or t[1]=='NNP' or t[1]=='NNS' or t[1]=='NNPS'])

    data['Nouns']

    to_be_lemmatized = []

    for nouns_per_medical_specialties in data['Nouns']:
            
        words2lemmatied = []
        for word in nouns_per_medical_specialties:
            words2lemmatied.append(word[0])
        
        to_be_lemmatized.append(words2lemmatied)

        data['to_be_lemmatized'] = to_be_lemmatized
        data['to_be_lemmatized']

    lmtzr = WordNetLemmatizer()

    # print("rocks :", lmtzr.lemmatize("rocks"))

    # for dd in data['to_be_lemmatized']:
    #     lemma = lmtzr.lemmatize(dd[0])
    #     print("%s Lemma:%s" %(dd[0], lemma))
    data['lemmatize'] = data['to_be_lemmatized'].apply(lambda lst:[lmtzr.lemmatize(word) for word in lst])
    print(data['lemmatize'])

    # data['to_be_lemmatized'].apply(lambda lst:[lmtzr.lemmatize(word) for word in lst])

    data['lemmatize_count'] = data['lemmatize'].astype('str')
    data['lemmatize_count']=data['lemmatize_count'].str.split().str.len()
    del data['to_be_lemmatized']
    del data['sample_name']

    data.reset_index(drop=True)
    #TODO Complete the function

    return data


if __name__ == "__main__":

    #get user folder
    user_folder = os.path.expanduser("~")
    csv_file = user_folder + "/Documents/MT/data.csv"
    df = read_data(csv_file)
    print(df.head())
    clean_data(df)
