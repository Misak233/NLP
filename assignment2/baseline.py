import nltk
import re
from nltk.corpus import stopwords
import os
import json
import requests
from collections import defaultdict
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix

data_dir = "" #'rumour-data'
stopwords = set(stopwords.words('english'))

def get_tweet_text_from_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data
def write_result(file_path,files):
    with open(file_path,'w',encoding='utf-8') as json_file:
        json.dump(files,json_file,ensure_ascii=False, indent=4)

tt =nltk.tokenize.regexp.WordPunctTokenizer()
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
def preprocess_events(events):
    final_event=[]
    for event in events:
        BOW={}
        event=removePunctuation(event)
        for sentences in event.split():
            sentences=tt.tokenize(sentences)
            for words in sentences:
                words=lemmatizer.lemmatize(words)
                if words not in stopwords:
                    BOW[words.lower()]=BOW.get(words,0)+1
        final_event.append(BOW)
    return final_event

train_set=get_tweet_text_from_json(os.path.join(data_dir,"train.json"))
train=[]
train_label=[]
for event in train_set:
        trains=train_set[event]["text"]
        train.append(trains)
        train_label.append(1)
train_set2=get_tweet_text_from_json(os.path.join(data_dir,"train2.json"))
for event in train_set2["text"]:
    trains=event
    train.append(trains)
    train_label.append(0)
punctuation = '!,;:?"\'.\''
def removePunctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text.strip()

transformer = TfidfTransformer(smooth_idf=False,norm=None)
train=preprocess_events(train)
vectorizer = DictVectorizer()
train_matrix = vectorizer.fit_transform(train)
train_matrix= transformer.fit_transform(train_matrix)

#development data 
dev_set=get_tweet_text_from_json(os.path.join(data_dir,"dev.json"))
dev=[]
dev_label=[]
for event in dev_set:
    devs=dev_set[event]["text"]
    dev.append(devs)
    if dev_set[event]["label"]==1:
        dev_label.append(1)
    if dev_set[event]["label"]==0:
        dev_label.append(0)
dev=preprocess_events(dev)
dev_matrix = vectorizer.transform(dev)
dev_matrix= transformer.transform(dev_matrix)

from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
def test_model(algorithm,train_data,train_classification,test_data,test_classification):
        algorithm.fit(train_data,train_classification)
        prediction=algorithm.predict(test_data)
        accuracy=accuracy_score(test_classification,prediction)
        report=classification_report(test_classification,prediction)
        print(accuracy)
        print(report)

#predictions = model_selection.cross_val_predict(MultinomialNB(), train_matrix,train_label, cv=10)
#print (classification_report(train_label,predictions))
print("Here is MultinomialNB")
test_model(MultinomialNB(),train_matrix,train_label,dev_matrix,dev_label)
#word order preserved with this architecture
print("Here is SVM")
test_model(SVC(gamma=2, C=1),train_matrix,train_label,dev_matrix,dev_label)
print("Here is LogisticRegression")
test_model(LogisticRegression(),train_matrix,train_label,dev_matrix,dev_label)



