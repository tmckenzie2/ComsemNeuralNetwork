import pickle
from tensorflow import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


# default classifier
sv_classifier = 0
tense_classifier = 0
noun_phrase_classifier = 0
final_classifier = "this sentence contains no errors"

# loading sv tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# loading sv neural network
new_model = keras.models.load_model('pickled_binary_nn')

# example input text
text = "The teenager should talk with parents and friend about what teenager want to be, want to do."

# text must be placed in an array in order to be manipulated by tokenzier
text_array = [text]

#tokenize text
tokens = tokenizer.texts_to_sequences(text_array)

#add padding to text to compare the text properly
tokens_pad = pad_sequences(tokens,maxlen=39,
                           padding='pre', truncating='pre')
tokens_pad.shape

#print example of confidence that there is a sv agreement error in text
print(new_model.predict(tokens_pad)[0][0])

# less than 0.5 means it contains no error, greater than 0.5 means it does contain sv error
if new_model.predict(tokens_pad)[0][0] > 0.5:
    sv_classifier =  new_model.predict(tokens_pad)[0][0]

#print the sv classifier
print("subject verb agreement error confidence ",sv_classifier)

#open tense error tokenizer
with open('tense_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#load tense error binary neural network
new_model = keras.models.load_model('pickled_tense_nn')

tokens = tokenizer.texts_to_sequences(text_array)

tokens_pad = pad_sequences(tokens,maxlen=39,
                           padding='pre', truncating='pre')
tokens_pad.shape

#print example of confidence that there is a tense agreement error in text
print(new_model.predict(tokens_pad)[0][0])

# less than 0.5 means it contains no error, greater than 0.5 means it may contain tense error
if new_model.predict(tokens_pad)[0][0] > 0.5:
    tense_classifier =  new_model.predict(tokens_pad)[0][0]

#print the tense classifier
print("tense error confidence level ", tense_classifier)

# open noun phrase tokenizer
with open('np_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# load noun phrase binary neural network
new_model = keras.models.load_model('pickled_np_nn')

tokens = tokenizer.texts_to_sequences(text_array)

tokens_pad = pad_sequences(tokens,maxlen=46,
                           padding='pre', truncating='pre')
tokens_pad.shape

#print example of confidence that there is a noun phrase error in text
print(new_model.predict(tokens_pad)[0][0])

# less than 0.5 means it contains no error, greater than 0.5 means it may contain noun phrase error
if new_model.predict(tokens_pad)[0][0] > 0.5:
    noun_phrase_classifier =  new_model.predict(tokens_pad)[0][0]

#print the noun phrase classifier
print("noun phrase confidence ", noun_phrase_classifier)

# compare confidence of each classifier and determine which has the highest confidence value
if (sv_classifier > tense_classifier) and (sv_classifier > noun_phrase_classifier):
   final_classifier = "this sentence has a subject verb agreement error"
elif (tense_classifier > sv_classifier) and (tense_classifier > noun_phrase_classifier):
   final_classifier = "this sentence has a tense error"
elif (noun_phrase_classifier > sv_classifier) and (noun_phrase_classifier > tense_classifier):
   final_classifier = "this sentence has a noun phrase error"
else:
    final_classifier = final_classifier

#print final classifer
print(final_classifier)
    





