import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from collections import defaultdict
import os
import json
import requests
from pathlib import Path

data_dir = "/Users/zxj/Desktop/study/semester2/NLP/rumour-data" #'rumour-data'
tt = TweetTokenizer()
stopwords = set(stopwords.words('english'))


def get_tweet_text_from_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data["text"]
    
def get_events(event_dir):
    event_list = []
    for event in sorted(os.listdir(event_dir)):
        ###
        # Your answer BEGINS HERE
        ###
        a=""
        dir=(os.path.join(event_dir,event))
        for home,dirs,files in os.walk(dir):
            for filename in files:
                a=a+(get_tweet_text_from_json(os.path.join(home,filename)))
        event_list.append(a)

        ###
        # Your answer ENDS HERE
        ###
        
    return event_list


def preprocess_events(events):
    ###
    # Your answer BEGINS HERE
    ###
    final_event=[]
    for event in events:
        BOW={}
        for sentences in event.split():
            sentences=tt.tokenize(sentences)
            for words in sentences:
                if words not in stopwords:
                    BOW[words]=BOW.get(words,0)+1
        final_event.append(BOW)
    return final_event
    ###
    # Your answer ENDS HERE
    ###
rumour_events = get_events(os.path.join(data_dir, "rumours"))
nonrumour_events = get_events(os.path.join(data_dir, "non-rumours"))

preprocessed_rumour_events = preprocess_events(rumour_events)
preprocessed_nonrumour_events = preprocess_events(nonrumour_events)

print(preprocessed_rumour_events[1])

print("Number of preprocessed rumour events =", len(preprocessed_rumour_events))
print("Number of preprocessed non-rumour events =", len(preprocessed_nonrumour_events))
