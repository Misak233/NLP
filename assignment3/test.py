import nltk
from nltk.corpus import brown
from nltk.corpus import wordnet

nltk.download("brown")
nltk.download("wordnet")

# filtered_gold_standard stores the word pairs and their human-annotated similarity in your filtered test set
filtered_gold_standard = {}

# lemmatizer
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma


###
# Your answer BEGINS HERE
###
N=brown.paras()
brown_corpus=[]
for i in N:
	x=[]
	for j in i:
		for k in j:
			if k.isalpha():
				k=k.lower()
				k=lemmatize(k)
				x.append(k)
	brown_corpus.append(x)

BOW={}
for i in brown_corpus:
    set1=[]
    for j in i:
        if j not in set1:
            BOW[j]=BOW.get(j,0)+1
        set1.append(j)
f=open('set1.tab')
point=1
for i in f.readlines():
	if point>1:
		k=1
		for j in i.split():
			if k==1:
				word1=j
			if k==2:
				word2=j
			if k==3:
				pro=j
				if word1 in BOW.keys() and word2 in BOW.keys():
					if BOW[word1]>=8 and BOW[word2]>=8:
						filtered_gold_standard[(word1,word2)]=filtered_gold_standard.get((word1,word2),0)+float(pro)
			k=k+1	
	point=point+1

###
# Your answer ENDS HERE
###
print(len(filtered_gold_standard))
print(filtered_gold_standard)

# final_gold_standard stores the word pairs and their human-annotated similarity in your final filtered test set
final_gold_standard = {}
word_primarysense = {} #a dictionary of (word, primary_sense) (used for next section); primary_sense is a synset

###
# Your answer BEGINS HERE
###
filtered_gold_standard_corpus=[]
for (i,j) in filtered_gold_standard.keys():
    if i not in filtered_gold_standard_corpus:
        filtered_gold_standard_corpus.append(i)
    if j not in filtered_gold_standard_corpus:
        filtered_gold_standard_corpus.append(j)        
for i in filtered_gold_standard_corpus:
    syns=wordnet.synsets(i)
    best_score=0
    next_score=0
    most_commen=[]
    for j in syns:
        count=0
        for k in j.lemmas():
            word=k.name()
            word=word.lower()
            word=lemmatize(word)
            if word==i:
                count=count+k.count()        
        if count>=best_score:
            next_score=best_score
            best_score=count
            most_commen=j
        elif count>=next_score:
            next_score=count
    if most_commen.pos()==wordnet.NOUN and best_score>=(4*next_score):
        word_primarysense[i]=word_primarysense.get(i,0)+0
        word_primarysense[i]=most_commen

for (i,j) in filtered_gold_standard.keys():
	if i in word_primarysense.keys() and j in word_primarysense.keys():
		final_gold_standard[(i,j)]=final_gold_standard.get((i,j),0)+filtered_gold_standard[(i,j)]

###
# Your answer ENDS HERE
###

print(len(final_gold_standard))
print(final_gold_standard)

from nltk.corpus import wordnet_ic
nltk.download('wordnet_ic')

# lin_similarities stores the word pair and Lin similarity mappings
lin_similarities = {}

###
# Your answer BEGINS HERE
###
brown_ic = wordnet_ic.ic('ic-brown.dat')
for (i,j) in final_gold_standard.keys():
	sim=word_primarysense[i].lin_similarity(word_primarysense[j],brown_ic)
	lin_similarities[(i,j)]=lin_similarities.get((i,j),0)+sim

###
# Your answer ENDS HERE
###

print(lin_similarities)

# NPMI_similarities stores the word pair and NPMI similarity mappings
NPMI_similarities = {}

###
# Your answer BEGINS HERE
###
import math
for (i,j) in final_gold_standard.keys():
	word1_count=0
	word2_count=0
	both_count=0
	total_count=0.0
	NPMI=0
	for x in brown_corpus:
		set1=[]
		for y in x:
			if y not in set1:
				total_count=total_count+1
				if y==i:
					word1_count+=1
				if y==j:
					word2_count+=1
			set1.append(y)
		if i in set1 and j in set1:
			both_count+=1
	if both_count!=0:		
		NPMI=(math.log((word1_count/total_count)*(word2_count/total_count),2)/math.log((both_count/total_count),2))-1
	if both_count==0:
		NPMI=-1
	NPMI_similarities[(i,j)]=NPMI_similarities.get((i,j),0)+NPMI

###
# Your answer ENDS HERE
###

print(NPMI_similarities)


# LSA_similarities stores the word pair and LSA similarity mappings
LSA_similarities = {}

###
# Your answer BEGINS HERE
###
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np
from numpy.linalg import norm
docu_term=[]
words=[]
for (i,j) in final_gold_standard:
	if i not in words:
		words.append(i)
	if j not in words:
		words.append(j)

for i in brown_corpus:
	bags={}
	for j in words:
		if j in i:
			bags[j]=bags.get(j,0)+1
		if j not in i:
			bags[j]=bags.get(j,0)+0
	docu_term.append(bags)

vectorizer = DictVectorizer()
matrix = vectorizer.fit_transform(docu_term)
svd = TruncatedSVD(n_components=500)
term_docu = csr_matrix(matrix).transpose()
result = svd.fit_transform(term_docu)
def sim_cos(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))

for (i,j) in final_gold_standard.keys():
    word1=result[vectorizer.vocabulary_[i]]
    word2=result[vectorizer.vocabulary_[j]]
    LSA_similarities[(i,j)]=LSA_similarities.get((i,j),0)+sim_cos(word1,word2)

x1=result[vectorizer.vocabulary_["professor"]]
x2=result[vectorizer.vocabulary_["doctor"]]  
print(x1)
###
# Your answer ENDS HERE
###

print(LSA_similarities[('professor', 'doctor')] > 0 and LSA_similarities[('professor', 'doctor')] < 0.4)
print(LSA_similarities)

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
    set1=[]
    for j in i:
        if j not in set1:
            BOW1[j]=BOW1.get(j,0)+1
        set1.append(j)
vocab1=[]
for i in corpus:
    for j in i:
        if j not in vocab1 and BOW1[j]>=5:
            vocab1.append(j)
vocab=vocab1

###
# Your answer ENDS HERE
###
print(len(vocab))