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

from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, GlobalMaxPooling1D
vocab_size=train_matrix.shape[1]
print(vocab_size)
train_matrix1=train_matrix[1069:1269]
train_label1=train_label[1069:1269]
model = Sequential(name="feedforward-bow-input")
model.add(layers.Dense(32, input_dim=vocab_size, activation='relu'))
#model.add(layers.Dense(22, input_dim=vocab_size, activation='relu'))
#model.add(layers.Dense(20, input_dim=vocab_size, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#since it's a binary classification problem, we use a binary cross entropy loss here
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(train_matrix,train_label, epochs=30, verbose=True,validation_data=(train_matrix1, train_label1), batch_size=22)
model.summary()
loss, accuracy = model.evaluate(dev_matrix, dev_label, verbose=False)
print("\nTesting Accuracy:  {:.4f}".format(accuracy))
result=model.predict(dev_matrix)
i=0
for event in dev_set:
    if result[i]>=0.5:
        dev_set[event]["label"]=1
    if result[i]<0.5:
        dev_set[event]["label"]=0
    i=i+1
write_result(os.path.join(data_dir,"dev-baseline-r.json"),dev_set)







