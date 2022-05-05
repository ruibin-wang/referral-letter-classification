import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, GlobalMaxPool1D, Conv1D
from tensorflow.keras.layers import Embedding
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import utils
from function_sets import plot_graphs



## Building the neural network model
def simple_neural_network(X_train, X_test, y_train, y_test, num_labels):
    embedding_dim = 50

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=30,
                        validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")



def add_droupout_regularization(X_train, X_test, y_train, y_test, num_labels):
    embedding_dim = 50

    model2 = Sequential()
    model2.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    model2.add(Flatten())
    model2.add(Dense(16, activation='relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(num_labels, activation='softmax'))

    model2.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
    model2.summary()

    history_two = model2.fit(X_train, y_train,
                             batch_size=32,
                             epochs=80,
                             validation_data=(X_test, y_test))

    loss, accuracy = model2.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model2.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_graphs(history_two, "accuracy")
    plot_graphs(history_two, "loss")



def add_maxpolling(X_train, X_test, y_train, y_test, num_labels):
    embedding_dim = 50

    model3 = Sequential()
    model3.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    model3.add(GlobalMaxPool1D())
    model3.add(Dense(16, activation='relu'))
    model3.add(Dense(num_labels, activation='softmax'))

    model3.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    model3.summary(X_train, X_test, y_train, y_test)

    history_3 = model3.fit(X_train, y_train,
                           batch_size=32,
                           epochs=80,
                           validation_data=(X_test, y_test))

    loss, accuracy = model3.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model3.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_graphs(history_3, "accuracy")
    plot_graphs(history_3, "loss")



def add_dropout_maxpooling(X_train, X_test, y_train, y_test, num_labels):
    embedding_dim = 50

    model4 = Sequential()
    model4.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    model4.add(GlobalMaxPool1D())
    model4.add(Dropout(0.2))
    model4.add(Dense(16, activation='relu'))
    model4.add(Dropout(0.2))
    model4.add(Dense(num_labels, activation='softmax'))

    model4.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    model4.summary()

    history_4 = model4.fit(X_train, y_train,
                           batch_size=32,
                           epochs=100,
                           validation_data=(X_test, y_test))

    loss, accuracy = model4.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model4.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_graphs(history_4, "accuracy")
    plot_graphs(history_4, "loss")




def CNN(X_train, X_test, y_train, y_test, num_labels):
    embedding_dim = 50

    model5 = Sequential()
    model5.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    model5.add(Conv1D(128, num_labels, activation='relu'))
    model5.add(GlobalMaxPool1D())

    # model5.add(Dense(64, activation='relu'))
    model5.add(Dense(64, activation='relu'))
    model5.add(Dense(num_labels, activation='softmax'))

    model5.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    model5.summary()

    history_5 = model5.fit(X_train, y_train,
                           batch_size=10,
                           epochs=80,
                           validation_data=(X_test, y_test))

    loss, accuracy = model5.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model5.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_graphs(history_5, "accuracy")
    plot_graphs(history_5, "loss")



if __name__ == "__main__":

    # df = pd.read_csv('./data/symp_diagnosis_relation_training.csv')
    # df = pd.read_csv('./data/complaint_training_data.csv')
    df = pd.read_csv('./data/combine_complaint_symp.csv')

    sentences = df['symptoms']
    y = df['diagnosis']

    sentences_train, sentences_test, train_y, test_y = train_test_split(sentences, y, test_size=0.25, random_state=42)


    tokenize = Tokenizer(num_words=1000)
    tokenize.fit_on_texts(sentences)
    # tokenize.fit_on_texts(sentences_train)

    X_train = tokenize.texts_to_sequences(sentences_train)
    X_test = tokenize.texts_to_sequences(sentences_test)

    vocab_size = len(tokenize.word_index) + 1

    maxlen = 100

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    encoder = LabelEncoder()
    encoder.fit(train_y)
    y_train = encoder.transform(train_y)
    y_test = encoder.transform(test_y)

    num_classes = np.max(y_train) + 1
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    num_labels = 4
    # simple_neural_network(X_train, X_test, y_train, y_test, num_labels)
    # add_droupout_regularization(X_train, X_test, y_train, y_test, num_labels)
    # add_maxpolling(X_train, X_test, y_train, y_test, num_labels)
    # add_dropout_maxpooling(X_train, X_test, y_train, y_test, num_labels)
    CNN(X_train, X_test, y_train, y_test, num_labels)


    print('11')







