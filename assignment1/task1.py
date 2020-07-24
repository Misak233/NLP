import json
import requests
import os
from pathlib import Path


data_dir = "/Users/zxj/Desktop/study/semester2/NLP/rumour-data" #'rumour-data'


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
    
#a list of events, and each event is a list of tweets (source tweet + reactions)    
rumour_events = get_events(os.path.join(data_dir, "rumours"))
nonrumour_events = get_events(os.path.join(data_dir, "non-rumours"))

print("Number of rumour events =", len(rumour_events))
print("Number of non-rumour events =", len(nonrumour_events))
