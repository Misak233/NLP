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
    i=0
    for event in events:
        BOW={}
        for sentences in event.split():
            sentences=tt.tokenize(sentences)
            for words in sentences:
                if words not in stopwords:
                    BOW[words]=BOW.get(words,0)+1
        final_event.append(BOW)
        i+=1
    return final_event
    ###
    # Your answer ENDS HERE
    ###
rumour_events = get_events(os.path.join(data_dir, "rumours"))
nonrumour_events = get_events(os.path.join(data_dir, "non-rumours"))

preprocessed_rumour_events = preprocess_events(rumour_events)
preprocessed_nonrumour_events = preprocess_events(nonrumour_events)



def get_all_hashtags(events):
    hashtags = set([])
    for event in events:
        for word, frequency in event.items():
            if word.startswith("#"):
                hashtags.add(word)
    return hashtags

hashtags = get_all_hashtags(preprocessed_rumour_events + preprocessed_nonrumour_events)


from nltk.corpus import wordnet

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
words = set(nltk.corpus.words.words()) #a list of words provided by NLTK


def tokenize_hashtags(hashtags):
    ###
    # Your answer BEGINS HERE
    ###
    hash_dic=defaultdict(list)
    word_length=max(len(word) for word in words)
    for hashtag in hashtags:
        tag_length=len(hashtag)
        while tag_length>0:
            cut_length=min(word_length,tag_length)
            for i in range(cut_length,0,-1):
                test_word=hashtag[tag_length-i:tag_length]
                test_word,tag=lemma(test_word)
                if test_word in words:
                    hash_dic[hashtag].append(test_word)
                    tag_length=tag_length-1
                    break
                elif i==1:
                    hash_dic[hashtag].append(test_word)
                    tag_length=tag_length-1
    return hash_dic

def lemma(word):
    word=word.lower()
    lem=lemmatizer.lemmatize(word,'v')
    if lem==word:
        lem=lemmatizer.lemmatize(word,'n')
    tag=lemma_tag(lem)
    return lem,tag

word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()
def lemma_tag(word):
    word=word_tokenizer.tokenize(word)
    pos_tags =nltk.pos_tag(word)
    return pos_tags

    ###
    # Your answer ENDS HERE
    ###


tokenized_hashtags = tokenize_hashtags(hashtags)

print(list(tokenized_hashtags.items())[:20])

print(len(tokenized_hashtags) == len(hashtags))

    





