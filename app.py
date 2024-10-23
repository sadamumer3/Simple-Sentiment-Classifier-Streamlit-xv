
import streamlit as st

import nltk
from nltk import word_tokenize
from nltk.classify import ClassifierI

from statistics import mode
import pickle

nltk.download('punkt_tab')

data_path = './data/'
models_path = './models/'
 
# get top 5000 words most used
import_word_feature = open(data_path+'all_pos_neg_features.pickle','rb')
word_features = pickle.load(import_word_feature)
import_word_feature.close()

def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	return features


# --- New Voting of multiple algorithmss---
#  based on merit or weighted voting 

class VoteClassifier(ClassifierI):
	def __init__(self,classifiers):
		self._classifiers = [c[0] for c in classifiers]
		self._accuracies = [c[1] for c in classifiers]
        # print(self._classifiers)
    
	def classify(self,features):
		votes = []
		for c,a in zip(self._classifiers, self._accuracies):
			v = c.classify(features)
			votes.append(v)
			# votes.append( (v, a) )
            
        # print(votes)
		return mode(votes)

	def conf(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		choice = votes.count( mode(votes) )
		conf = choice / len(votes)
		return conf


import_vt_classifier = open(models_path + 'vt_classifier.pickle','rb')
vt_classifier = pickle.load(import_vt_classifier)
import_vt_classifier.close()

# Define a simple function for sentiment analysis using the loaded model
def sentiment(text):
    features_found = find_features(text)
    prediction = vt_classifier.classify(features_found)
    confidence = vt_classifier.conf(features_found)
    return prediction, confidence

# Streamlit interface
st.title('Simple Sentiment Classifier xv')

# User input for movie review
user_input = st.text_area("Enter a movie review:")


# Analyze sentiment when button is clicked
if st.button("Analyze Sentiment"):
    prediction, confidence = sentiment(user_input)
    prediction = "Positive" if prediction=='pos' else 'Negative'
    st.write(f"Sentiment: {prediction}")
    st.write(f"Confidence: {confidence*100:.2f}%")


