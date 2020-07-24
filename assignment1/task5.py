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
def update_event_bow(events):
    ###
    # Your answer BEGINS HERE
    ###
    event_list=[]
    for event in events:
        new_event=defaultdict(list)
        for word,freq in event.items():
            if word in hashtags:
                word=tokenized_hashtags[word]
                for i in word:
                   new_event[i]=new_event.get(i,0)+freq
        event.update(new_event)
        event_list.append(event)
    return event_list
    ###
    # Your answer ENDS HERE
    ###
tokenized_hashtags = tokenize_hashtags(hashtags)
        
update_event_bow(preprocessed_rumour_events)
update_event_bow(preprocessed_nonrumour_events)



from sklearn.feature_extraction import DictVectorizer

vectorizer = DictVectorizer()

###
# Your answer BEGINS HERE
###
def get_data(rumours_data,nonrumour_data,ratio):
    train_data=[]
    train_classification=[]
    test_data=[]
    test_classification=[]
    rumours_length=len(rumours_data)
    nonrumour_length=len(nonrumour_data)
    cutoff1=int(ratio*rumours_length)
    cutoff2=int(ratio*nonrumour_length)
    for data in rumours_data[:cutoff1]:
        train_classification.append("rumours")
        train_data.append(data)
    for data in nonrumour_data[:cutoff2]:
        train_classification.append("non-rumours")
        train_data.append(data)
    for data in rumours_data[cutoff1:]:
        test_data.append(data)
        test_classification.append("rumours")
    for data in nonrumour_data[cutoff2:]:
        test_data.append(data)
        test_classification.append("non-rumours")
    train_data=vectorizer.fit_transform(train_data)
    test_data=vectorizer.transform(test_data)   
    return train_data,train_classification,test_data,test_classification
###
# Your answer ENDS HERE
###
train,train_class,test,test_class=get_data(preprocessed_rumour_events,preprocessed_nonrumour_events,[0.6,0.2])

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
###
# Your answer BEGINS HERE
###
def train_model(algorithms,train_data,train_classification,test_data,test_classification):
    for algorithm in algorithms:
        algorithm.fit(train_data,train_classification)
        prediction=algorithm.predict(test_data)
        accuracy=accuracy_score(test_classification,prediction)
        #report=classification_report(test_classification,prediction)
        print(accuracy)
        #print(report)

algorithm1=[]
a_test=[i for i in range(1,10)]
algorithm1=[MultinomialNB(alpha=a) for a in a_test]
print("Here is result of naive_bayes which range alpha from 1 to 10:")
train_model(algorithm1,train,train_class,test,test_class) #when alpha=2 the result is fairly better
algorithm2=[]
c_test=[0.01,0.1,1,10,100,1000]
print("Here is result of LogisticRegression whcih range C from 0.01 to 1000:")
algorithm2=[LogisticRegression(C=c) for c in c_test]#in this experiment, change of C wouldn't affect too much
train_model(algorithm2,train,train_class,test,test_class)
        



