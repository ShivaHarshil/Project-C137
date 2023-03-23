# Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# to stem words
from nltk.stem import PorterStemmer

# create an instance of class PorterStemmer

stemmer = PorterStemmer()
# importing json lib
import json
import pickle
import numpy as np

words=[] #list of unique roots words in the data
classes = [] #list of unique tags in the data
pattern_word_tags_list = [] #list of the pair of (['words', 'of', 'the', 'sentence'], 'tags')

# words to be ignored while creating Dataset
ignorewords = ['?', '!',',','.', "'s", "'m"]

# open the JSON file, load data from it.
train_data_file = open('intents.json')
data = json.load(train_data_file)
train_data_file.close()

# creating function to stem words
def get_stem_words(words, ignorewords):
    stemwords = []
    for word in words:

        # write stemming algorithm:
        '''
        Check if word is not a part of stop word:
        1) lowercase it 
        2) stem it
        3) append it to stem_words list
        4) return the list
        ''' 
        # Add code here # 
        if word not in ignorewords:
            w = stemmer.stem(word.lower())
            stemwords.append(w)
        print(stemwords)       

    return stemwords


'''
List of sorted stem words for our dataset : 

['all', 'ani', 'anyon', 'are', 'awesom', 'be', 'best', 'bluetooth', 'bye', 'camera', 'can', 'chat', 
'cool', 'could', 'digit', 'do', 'for', 'game', 'goodby', 'have', 'headphon', 'hello', 'help', 'hey', 
'hi', 'hola', 'how', 'is', 'later', 'latest', 'me', 'most', 'next', 'nice', 'phone', 'pleas', 'popular', 
'product', 'provid', 'see', 'sell', 'show', 'smartphon', 'tell', 'thank', 'that', 'the', 'there', 
'till', 'time', 'to', 'trend', 'video', 'what', 'which', 'you', 'your']

'''


# creating a function to make corpus
def create_bot_corpus(words, classes, pattern_word_tags_list, ignorewords):

    for intent in data['intents']:

        # Add all patterns and tags to a list
        for pattern in intent['patterns']:  

            # tokenize the pattern          
            pattern_words = nltk.word_tokenize(pattern)
            words.extend(pattern_word)                      
            pattern_word_tags_list.append((pattern_word, intent['tag']))
        # Add all tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            stem_words = get_stem_words(words, ignorewords)


print(stemwords)
print(pattern_word_tags_list[0]) 
print(classes)   


def create_bot_corpus(stem_words, classes):

    stemwords = sorted(list(stem_words))
    classes = sorted(list(set(classes)))

    pickle.dump(stemwords, open('words.pkl','wb'))
    pickle.dump(classes, open('classes.pkl','wb'))

    return stem_words, classes

stemwords, classes = create_bot_corpus(stemwords,classes)  

print(stemwords)
print(classes)     
    
 




def bag_of_words_encoding(stemwords, pattern_word_tags_list):
    
    bag = []
    for word_tags in pattern_word_tags_list:
        # example: word_tags = (['hi', 'there'], 'greetings']

        pattern_words = word_tags[0] # ['Hi' , 'There]
        bag_of_words = []

        # stemming pattern words before creating Bag of words
        stemmed_pattern_word = get_stem_words(pattern_words, ignorewords)

        # Input data encoding 
        '''
        Write BOW algo :
        1) take a word from stem_words list
        2) check if that word is in stemmed_pattern_word
        3) append 1 in BOW, otherwise append 0
        '''
        
        bag.append(bag_of_words)
    
    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
    
    labels = []

    for word_tags in pattern_word_tags_list:

        # Start with list of 0s 
        labels_encoding = list([0]*len(classes))  

        # example: word_tags = (['hi', 'there'], 'greetings']

        tag = word_tags[1]   # 'greetings'

        tag_index = classes.index(tag)

        # Labels Encoding
        labels_encoding[tag_index] = 1

        labels.append(labels_encoding)
        
    return np.array(labels)

