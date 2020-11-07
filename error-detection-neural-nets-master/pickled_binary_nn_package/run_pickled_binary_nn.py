import pickle
from tensorflow import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


# default classifier
classifier = "contains no error"

# loading tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# loading neural network
new_model = keras.models.load_model('pickled_binary_nn')

# example input text
text = "My uncle give me a lot of space."

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
    classifier = "contains subject verb agreement error"

#print the classifier
print(classifier)



