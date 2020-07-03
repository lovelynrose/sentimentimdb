import types
import time
import nltk
import spacy

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)

from data.parameters.datasetnames import PROCESSED
from sentiment.interface.readdata import ReadData
from sentiment.interface.savedata import SaveData


class Lemmatize():

    def __init__(self, lis=None, res=None, inptext=None, func=None, type='spacy', choice='imdb'):
        self.lis = lis
        self.res = res
        self.text = inptext
        self.choice = choice
        self.type = type
        if func is not None:
            self.execute = types.MethodType(func, self)

    def nltk_tag_to_wordnet_tag(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def process_list(self):
        lis1 = []
        start_time = time.perf_counter()
        print("Lemmatizing...")
        for i in range(len(self.lis)):
            self.text = self.lis[i]
            if self.type == 'nltk':
                lem = Lemmatize(using_nltk(self))  # Give spacy if that is to be used
            else:
                lem = Lemmatize(using_spacy(self))
            lis1.append(self.res)
            del lem
        end_time = time.perf_counter()
        print("Completed Lemmatizing : ", start_time - end_time)
        self.res = lis1

    def perform_expand(self, filename):
        get = ReadData(filename, choice=self.choice)
        self.lis = get.readDataReview('0')

        #print("Before lemmatize", filename, self.lis)
        #self.res = self.lis
        self.process_list()
        SaveData(PROCESSED['lemmatize'], choice=self.choice).saveDataReview(self.res)

    def lemmatize_text(self):
        if self.type == 'spacy':
            text = nlp(self.text)
            text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
            return text
        else:
            # tokenize the sentence and find the POS tag for each token
            nltk_tagged = nltk.pos_tag(nltk.word_tokenize(self.text))
            # tuple of (token, wordnet_tag)
            wordnet_tagged = map(lambda x: (x[0], self.nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
            lemmatized_sentence = []
            for word, tag in wordnet_tagged:
                if tag is None:
                    # if there is no available tag, append the token as is
                    lemmatized_sentence.append(word)
                else:
                    # else use the tag to lemmatize the token
                    lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
            return " ".join(lemmatized_sentence)


    def execute(self):
        pass

def using_nltk(self):
    # tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(self.text))
    # tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], self.nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    # print(lemmatized_sentence)
    self.res = " ".join(lemmatized_sentence)


def using_spacy(self):
    text = nlp(self.text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    self.res = text


