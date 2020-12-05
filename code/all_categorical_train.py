import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, Flatten
from tensorflow.python.keras.optimizers import Adam
#from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import backend as K

PATH_TO_DATA = '../data/processed/'
PATH_TO_TRAINING_DATA = PATH_TO_DATA + 'all_train_categorical.csv'
PATH_TO_TEST_DATA = PATH_TO_DATA + 'all_test_categorical.csv'
NUM_ERRORS = 4
NUM_WORDS = 10000

tf.compat.v1.disable_eager_execution()


def load_data_from_file(file_name):
    text = []
    labels = []

    with open(file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            text.append(row[0])
            labels.append(float(row[3]))

    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_ERRORS)
    return text, one_hot_labels


def tokenize_text(train_set, test_set):
    combined = train_set + test_set
    tokenizer = Tokenizer(num_words=NUM_WORDS)

    tokenizer.fit_on_texts(combined)
    train_tokens = tokenizer.texts_to_sequences(train_set)
    test_tokens = tokenizer.texts_to_sequences(test_set)

    return train_tokens, test_tokens


def prepare_data():
    x_train, y_train = load_data_from_file(PATH_TO_TRAINING_DATA)
    x_test, y_test = load_data_from_file(PATH_TO_TEST_DATA)
    x_train_tokens, x_test_tokens = tokenize_text(x_train, x_test)

    max_tokens = max([len(tokens) for tokens in x_train_tokens + x_test_tokens])

    x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)
    x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)

    return x_train_pad, y_train, x_test_pad, y_test, max_tokens


def build_model(max_tokens):
    model = Sequential()
    embedding_size = 8
    optimizer = Adam(lr=1e-3)

    model.add(Embedding(input_dim=NUM_WORDS, output_dim=embedding_size, input_length=max_tokens, name='layer_embedding'))
    model.add(GRU(units=16, return_sequences=True))
    model.add(GRU(units=8, return_sequences=True))
    model.add(GRU(units=4))
    model.add(Dense(NUM_ERRORS, activation='softmax')) # changed ativation to softmax and first parameter from one to 2 vhl

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) # changed loss to categorical_crossentropy

    return model


def main():
    x_train, y_train, x_test, y_test, max_tokens = prepare_data()
    model = build_model(max_tokens)

    print("training model")

    model.fit(x_train, y_train, validation_split=0.05, epochs=3, batch_size=32) # replaced y_train with one_hot_labels 
    result = model.evaluate(x_test, np.array(y_test)) # replaced y_test with two_hot_abels

    print("Accuracy: {0:.2%}".format(result[1]))


if __name__ == "__main__":
    main()