# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 20:46:41 2021

@author: chikk
"""


#import nltk
#nltk.download('vader_lexicon')
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#import tweets
tweets = pd.read_excel('cleveland clinic tweets.xlsx')
print (tweets)

#call the function
sia = SentimentIntensityAnalyzer()

#apply sia and transform them into the dataframe
tweets['neg'] = tweets['Data_field'].apply(lambda x:sia.polarity_scores(x)['neg'])
tweets['neu'] = tweets['Data_field'].apply(lambda x:sia.polarity_scores(x)['neu'])
tweets['pos'] = tweets['Data_field'].apply(lambda x:sia.polarity_scores(x)['pos'])
tweets['compound'] = tweets['Data_field'].apply(lambda x:sia.polarity_scores(x)['compound'])

pos_tweets = [j for i, j in enumerate(tweets['Data_field']) if tweets['compound'][i] > 0.2]
neu_tweets = [j for i, j in enumerate(tweets['Data_field'])if 0.2>=tweets['compound'][i]>=-0.2]
neg_tweets = [j for i, j in enumerate(tweets['Data_field'])if tweets['compound'][i]< -0.2]

print()


print ("percentage of positive tweets:{}%".format(len(pos_tweets)*100/len(tweets['Data_field'])))
print ("percentage of neural tweets:{}%".format(len(neu_tweets)*100/len(tweets['Data_field'])))
print ("percentage of negative tweets:{}%".format(len(neg_tweets)*100/len(tweets['Data_field'])))

#sample reuslt
print(tweets.head())

from textblob import TextBlob
tweepy = str(tweets)
ob = TextBlob(tweepy)
print(ob.sentiment.polarity)         
print(ob.sentiment.subjectivity)    
print(ob.sentiment)


