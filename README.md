# Comsem Neural Network

Note: Before you run any of this code make sure to pip install tensorflow and any other modules you may not have..

The entire training model will look like this
```python
import pickle
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from scipy.spatial.distance import cdist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import backend as K



tf.compat.v1.disable_eager_execution()

x_train_text = []
y_train = []
x_test_text = []
y_test = []

#read in training set that the neural network will be trained on
with open('binary_train_set.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        #declare which row contains the labels for the sentences in this
        #case the labels will be subject verb agreement error - 1, and
        #correct sentences - 0
        one_hot_labels = tf.keras.utils.to_categorical(row[3], num_classes=2)
        #place data into arrays
        x_train_text.append(row[0])
        y_train.append(float(row[3]))
        
#do the same thing with the test set
with open('binary_test_set.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x_test_text.append(row[0])
        y_test.append(float(row[3]))

#combine data so that you can tokenize every word in each set which means
#that each word will have a vector representation of numbers
data_text = x_train_text + x_test_text

#maximum number of words that will be tokenized
num_words=10000
tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(data_text)

#tokenize the training set and test set
x_train_tokens = tokenizer.texts_to_sequences(x_train_text)
x_test_tokens = tokenizer.texts_to_sequences(x_test_text)

num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

max_tokens = np.max(num_tokens)
max_tokens = int(max_tokens)

#pre pad the vector sequences with 0's so that each sequence is of the same
#length
pad = 'pre'

x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)


idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))

def tokens_to_string(tokens):
    # Map from tokens back to words.
    words = [inverse_map[token] for token in tokens if token != 0]
    
    # Concatenate all words.
    text = " ".join(words)

    return text

model = Sequential()

#the embedding layer will be of size 16 so a vector of x by 16
embedding_size = 16

model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding'))

#The model will have 3 layers consisting of Gated Recurrent Units
#GRU's are similar to Long Short Term Memory Units, they keep track
#of the hidden state within a sequence so that the hidden state contains
#more data about the beginning of the sentence and can apply that data to
#the end of a sentence to better keep track of grammar within an expression
model.add(GRU(units=16, return_sequences=True))
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))

#apply a sigmoid activation function to interpret the derivative of the loss function
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

#The dataset is read through 3 times (epochs) and is read through in batches of
#64 expressions at a time
model.fit(x_train_pad, y_train,
          validation_split=0.05, epochs=3, batch_size=64)

result = model.evaluate(x_test_pad, np.array(y_test))

print("Accuracy: {0:.2%}".format(result[1]))
print("-----------------------------------------------------")

y_pred = model.predict(x=x_test_pad[0:1000])
y_pred = y_pred.T[0]

cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred])
cls_true = np.array(y_test[0:1000])

incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]
```

## Training Set and Test Set
This model will be trained using a training set of sentences with grammatical errors and without grammatical errors. Since this is a binary model the errors will be classified in binary with a 0 representing the grammatically correct sentences and 1 representing the grammatically incorrect sentences. If you view the csv you can see that the sentences are included in the first column of the csv and their classifiers are included in the fourth column of the csv. You can test the accuracy of the model after it is trained on the training set by including a mix of correct and incorrect sentences in your test set. This portion of the model is where you will specify which training and test set you will be running the model with. 

```python
#read in training set that the neural network will be trained on
with open('binary_train_set.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        #declare which row contains the labels for the sentences in this
        #case the labels will be subject verb agreement error - 1, and
        #correct sentences - 0
        one_hot_labels = tf.keras.utils.to_categorical(row[3], num_classes=2)
        #place data into arrays
        x_train_text.append(row[0])
        y_train.append(float(row[3]))
        
#do the same thing with the test set
with open('binary_test_set.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x_test_text.append(row[0])
        y_test.append(float(row[3]))
```

Make sure you include the training set and test set csv files in the directory that your python file is in when you run the model. The csv's in this repo are kind of a mess but I will break them down with links here and show you what errors we tested and which files are the training set and which are the test set for that error --

subject verb agreement errors:
training set - https://github.com/tmckenzie2/ComsemNeuralNetwork/blob/main/error-detection-neural-nets-master/cleantrainset.csv
test set - https://github.com/tmckenzie2/ComsemNeuralNetwork/blob/main/error-detection-neural-nets-master/cleantestset.csv

tense errors:
training set - https://github.com/tmckenzie2/ComsemNeuralNetwork/blob/main/error-detection-neural-nets-master/cleantrainset_tense_binary.csv
test set - https://github.com/tmckenzie2/ComsemNeuralNetwork/blob/main/error-detection-neural-nets-master/cleantestset_tense_binary.csv

noun phrase errors:
training set - https://github.com/tmckenzie2/ComsemNeuralNetwork/blob/main/error-detection-neural-nets-master/NPtrain.csv
test set - https://github.com/tmckenzie2/ComsemNeuralNetwork/blob/main/error-detection-neural-nets-master/NPtest.csv


## Tokenizer
Before we pass our data into the input layer of the model we need to tokenize each word in the sentences included in our datasets. I will put a link to a tokenizer description here because they will probably do a much better job at describing what it does https://www.kdnuggets.com/2020/03/tensorflow-keras-tokenization-text-data-prep.html 

You need to tokenize these sentences so that your model can better understand the relationships and dependencies that words have to other words in each sentence in a way a computer understands best.. as numbers, specifically vectors.

```python
#combine data so that you can tokenize every word in each set which means
#that each word will have a vector representation of numbers
data_text = x_train_text + x_test_text

#maximum number of words that will be tokenized
num_words=10000
tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(data_text)

#tokenize the training set and test set
x_train_tokens = tokenizer.texts_to_sequences(x_train_text)
x_test_tokens = tokenizer.texts_to_sequences(x_test_text)

num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

max_tokens = np.max(num_tokens)
max_tokens = int(max_tokens)
```
## Embedding Layer
Again I will put a link to an article here because I feel like they do a much better job of explaining things https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

```python
#the embedding layer will be of size 16 so a vector of x by 16
embedding_size = 16

model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding'))
```

## Gated Recurrent Units
Note: To fully understand why these layers were set up make sure you understand the structure of Recurrent Neural Networks and Long Short Term Memory Cells
https://colah.github.io/posts/2015-08-Understanding-LSTMs/
https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be 
I tested the model using LSTM cells and Gated Recurrent Units (GRU) and GRU gave superior results.

```python
model.add(GRU(units=16, return_sequences=True))
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))
```

## Activation Function
Note: Make sure you understand the purpose of activation functions in the layers of your Neural Network
Many activation functions were tested for our output layer, Sigmoid was the best..

```python
#apply a sigmoid activation function to interpret the derivative of the loss function
model.add(Dense(1, activation='sigmoid'))
```

## Adam Optimizer
https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

```python
optimizer = Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
```

# Epochs + Batches
The number of epochs you include is how many times the model will run through your data and the batch size is the number of sentences that will be included every time your weights are adjusted. You may think that the more times your model runs through the data the better.. but you don't want to overfit your model because it will be too dependent on the training set to be able to apply the weights to new sentences.
https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
```python
model.fit(x_train_pad, y_train,
          validation_split=0.05, epochs=3, batch_size=32, shuffle = True,steps_per_epoch=None)

result = model.evaluate(x_test_pad, np.array(y_test))
```
After you run the model you will get an accuracy rating and a loss calculation for your model on the test set             
```
ms/sample - loss: 0.3414 - accuracy: 0.8696
Accuracy: 86.96%
```




