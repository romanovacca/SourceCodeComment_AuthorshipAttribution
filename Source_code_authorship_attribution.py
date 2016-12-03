import os
import nltk
import glob
import collections
import numpy as np
import tkMessageBox
from Tkinter import Tk
from collections import Counter
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten
from tkFileDialog import askdirectory
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

## globals
DOCUMENT_COUNTER = 0
SIMILARITIES = []
simcounter = 0
testcounter = 0

def load_data():
	global testcounter
	## import data into a list
	Tk().withdraw()
	directory = askdirectory()
	return_set = []
	for file in os.listdir(directory):
		if file.endswith(".txt"):
			
			raw_data_list = []
			processed_data_list = []
			return_list = []
			return_list.append(file)
			file = open(str(directory) + "/" + str(file))
			raw_data_list.append(file.read())
			for line in raw_data_list:
				line = line.split('\n')
				for strings in line:
					stripped = strings.strip()
					stripped = stripped.lower()
					processed_data_list.append(stripped)
			file.close()
			## select the comment lines and store them in another list
			comment_lines = []
			for pair_of_words in processed_data_list:
			  		if pair_of_words.startswith('#'):
			  			comment_lines.append(pair_of_words)
			  			
			if testcounter == 0:
				file = open("test100.txt", "w")
				## add the "." in front of "\n" so the lines can be seen as seperate sentences in the file
				file.write(".\n".join(str(i) for i in comment_lines))
				file.close()
			else: 
				 "niks"
			tokenized_sentences = [word_tokenize(i) for i in comment_lines]
			train = ' '.join(str(r) for v in tokenized_sentences for r in v)

			testcounter += 1
			return_list.append(train)
			return_set.append(return_list)
	return return_set

def calculate_similarity(train_set, test_set):
	## measure the tfidf
	global SIMILARITIES
	for element in train_set:
		for item in test_set:
			test_and_train = []
			test_and_train.append(element[1])
			test_and_train.append(item[1])
			#print test_and_train
			vect = TfidfVectorizer(min_df=1)
			tfidf = vect.fit_transform(test_and_train)
			cosine=(tfidf * tfidf.T).A
			tup = (str(item[0]), round((cosine[0][1])*100,3))
			SIMILARITIES.append(tup)

def get_highest_similarity():
	global SIMILARITIES
	sim = []
	for similarity in SIMILARITIES:
		sim.append(similarity[1])
	return sorted(sim)

data_folder = "../Thesis"
files = sorted(glob.glob(os.path.join(data_folder, "test*.txt")))
chapters = []
for fn in files:
    with open(fn) as f:
        chapters.append(f.read().replace('\n', ' '))
all_text = ' '.join(chapters)
feature_names = ["Lexical","Punctuation","Syntactic"]
feature_counter = 0


def PredictAuthors(fvs):
    """
    Use k-means clustering to fit a model
    """
    km = KMeans(n_clusters=2, init='k-means++', n_init=10, verbose=0)
    km.fit(fvs)

    return km

def LexicalFeatures():
    """
    Compute feature vectors for word and punctuation features
    """
    num_chapters = len(chapters)
    fvs_lexical = np.zeros((len(chapters), 4), np.float64)
  
    fvs_punct = np.zeros((len(chapters), 4), np.float64)
    for e, ch_text in enumerate(chapters):
        # note: the nltk.word_tokenize includes punctuation (it preserves the symbols)
        tokens = nltk.word_tokenize(ch_text.lower())
        words = word_tokenizer.tokenize(ch_text.lower()) ##### !!!!!!!!!!!!!!!!!!!!!!!!!!
        sentences = sentence_tokenizer.tokenize(ch_text)
        vocab = set(words)
        words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
                                       for s in sentences])
        if fvs_lexical[e, 0] >= 0:
            fvs_lexical[e, 0] = words_per_sentence.mean()
        else: 
            pass
        ## sentence length variation
        if fvs_lexical[e, 1] >= 0:
            fvs_lexical[e, 1] = words_per_sentence.std()
        else: 
            pass
        ## Lexical diversity, which is a measure of the richness of the authors vocabulary
        if fvs_lexical[e, 2] >= 0 and float(len(words)) > 0:
            fvs_lexical[e, 2] = len(vocab) / float(len(words))
        else:
            pass
        ## Number of different words per comment
        if fvs_lexical[e, 3] >= 0 and tokens.count('#') > 0:
            fvs_lexical[e, 3] = len(vocab) / tokens.count('#') 
        else:
            pass

        ## Semicolons per sentence
        fvs_punct[e, 0] = tokens.count(';') / float(len(sentences))
        ## Colons per sentence
        fvs_punct[e, 1] = tokens.count(':') / float(len(sentences))
        ## Commas per sentence
        fvs_punct[e, 2] = tokens.count(',') / float(len(sentences))
        ## Hashtags per sentence
        fvs_punct[e, 3] = tokens.count('#') / float(len(sentences))

    fvs_lexical = whiten(fvs_lexical)
    fvs_punct = whiten(fvs_punct)
    return fvs_lexical, fvs_punct

def SyntacticFeatures():
    """
    Extract feature vector for part of speech frequencies
    """
    def token_to_pos(ch):
        tokens = nltk.word_tokenize(ch)
        return [p[1] for p in nltk.pos_tag(tokens)]

    chapters_pos = [token_to_pos(ch) for ch in chapters]
    pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']
    fvs_syntax = np.array([[ch.count(pos) for pos in pos_list]
                           for ch in chapters_pos]).astype(np.float64)

    # normalise by dividing each row by number of tokens in the files
    fvs_syntax /= np.c_[np.array([len(ch) for ch in chapters_pos])]
    return fvs_syntax

def execute_code():
    global feature_counter, feature_names
    feature_sets = list(LexicalFeatures())
    feature_sets.append(SyntacticFeatures())
    classifications = [PredictAuthors(fvs).labels_ for fvs in feature_sets]
    for results in classifications:
        # in this case, we know the author of the last file, so set it to
        if results[15] == 0: results = 1 - results
        predictions = (','.join([str(a) for a in results]))
        print predictions + ''+ feature_names[feature_counter]
        feature_counter +=1


#############################
## Execute code
#############################
train_set = load_data()
tkMessageBox.showinfo("Directory selected", "Select next directory")
test_set = load_data()
calculate_similarity(train_set, test_set)
get_highest_similarity()
print SIMILARITIES
zero_and_one = []
for all_tup in SIMILARITIES:
	if SIMILARITIES[simcounter][1] == get_highest_similarity()[-2]:
		zero_and_one.append(1)
	else:
		zero_and_one.append(0)
	result = (','.join([str(a) for a in zero_and_one]))
	simcounter += 1
print result + '' +"Similarity"
execute_code()
