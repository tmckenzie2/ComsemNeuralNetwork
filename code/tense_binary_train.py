import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import backend as K


PATH_TO_TRAINING_DATA = '../data/processed/tense_train_binary.csv'
PATH_TO_TEST_DATA = '../data/processed/tense_test_binary.csv'
NUM_WORDS = 10000


tf.compat.v1.disable_eager_execution() 


def load_data_from_file(file_name):
    text = []
    labels = []

    with open(file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            one_hot_labels = tf.keras.utils.to_categorical(row[3], num_classes=2)
            text.append(row[0])
            labels.append(float(row[3]))

    return text, labels


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
    optimizer = Adam(lr=1e-3)
    embedding_size = 8

    model.add(Embedding(input_dim=NUM_WORDS, output_dim=embedding_size, input_length=max_tokens, name='layer_embedding'))
    model.add(GRU(units=16, return_sequences=True))
    model.add(GRU(units=8, return_sequences=True))
    model.add(GRU(units=4))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def print_results(model, text, labels):
    y_pred = model.predict(x=text[0:1000]).T[0]
    y_pred = y_pred.T[0]

    cls_pred = np.array([1.0 if p > 0.5 else 0.0 for p in y_pred])
    cls_true = np.array(labels[0:1000])

    incorrect = np.where(cls_pred != cls_true)
    incorrect = incorrect[0]

    for i in incorrect:
        print (text[i]) 


def main():
    x_train, y_train, x_test, y_test, max_tokens = prepare_data()
    model = build_model(max_tokens)

    print("training model")

    model.fit(x_train, y_train, validation_split=0.05, epochs=3, batch_size=32, shuffle = True,steps_per_epoch=None)
    result = model.evaluate(x_test, np.array(y_test))
    print("Accuracy: {0:.2%}".format(result[1]))


if __name__ == "__main__":
    main()