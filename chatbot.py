
# coding: utf-8

# # Meet ChatterBox:

import nltk
import warnings
warnings.filterwarnings("ignore")

# nltk.download() # for downloading packages

import numpy as np
import random
import io
import string # to process standard python strings


readFile=io.open('data.txt','r',errors = 'ignore')
data=readFile.read()
lowerData=data.lower()# converts to lowercase

#nltk.download('punkt') # first-time use only # for downloading packages
#nltk.download('wordnet') # first-time use only # for downloading packages


sentences_tokens = nltk.sent_tokenize(lowerData)# converts to list of sentences 
word_tokens = nltk.word_tokenize(lowerData)# converts to list of words


sentences_tokens[:2]


word_tokens[:5]


lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

INPUTS = ("hello", "hi", "greetings", "good morning", "good evening","good afternoon",)
RESPONSES = ["hi", "hello",]



# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in INPUTS:
            return random.choice(RESPONSES)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Generating response
def response(user_response):
    chatterBox_response=''
    sentences_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentences_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        chatterBox_response=chatterBox_response+"I am sorry! I am not able to get you"
        return chatterBox_response
    else:
        chatterBox_response = chatterBox_response+sentences_tokens[idx]
        return chatterBox_response


flag=True
print("ChatterBox: My name is ChatterBox. I will answer your queries about ChatterBox. If you want to exit, sat bye or thanks!")

while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ChatterBox: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ChatterBox: "+greeting(user_response))
            else:
                print("ChatterBox: ",end="")
                print(response(user_response))
                sentences_tokens.remove(user_response)
    else:
        flag=False
        print("ChatterBox: Bye! have a nice day..")    
        
        

