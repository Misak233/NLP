import nltk
from nltk.corpus import brown
from nltk.corpus import wordnet

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma

N=brown.paras()

num_train = 12000
UNK_symbol = "<UNK>"
vocab = set([UNK_symbol])

###
# Your answer BEGINS HERE
###
Num=1
corpus=[]
for i in N:
    x=[]
    for j in i:
        if Num<=num_train:
            for k in j:
                k=k.lower()
                x.append(k)
    corpus.append(x)
BOW1={}
for i in corpus:
    set=[]
    for j in i:
        if j not in set:
            BOW1[j]=BOW1.get(j,0)+1
        set.append(j)
vocab1=[]
for i in corpus:
    for j in i:
        if j not in vocab1 and BOW1[j]>=5:
            vocab1.append(j)
vocab=vocab1
###
# Your answer ENDS HERE

print(len(vocab1))
print(len(vocab))
	
