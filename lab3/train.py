#! /usr/bin/python3

from codemaps import *
from dataset import *
import numpy as np
import sys
from contextlib import redirect_stdout

from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate, Lambda
import tensorflow as tf

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


def build_network(codes):

    # sizes
    n_words = codes.get_n_words()
    n_wordsLC = codes.get_n_wordsLC()
    n_sufs = codes.get_n_sufs()
    n_prefs = codes.get_n_prefs()
    n_labels = codes.get_n_labels()
    n_external = codes.get_n_external()
    max_len = codes.maxlen

    w_embedding_dim = [50, 100, 200][-1]
    wLC_embedding_dim = [50, 100, 200][-1]
    np.random.seed(2)
    w_embedding_matrix = np.random.randn(n_words, w_embedding_dim)
    wLC_embedding_matrix = np.random.randn(n_wordsLC, wLC_embedding_dim)

    embeddings_index = dict()
    f = open('glove/glove.6B.' + str(w_embedding_dim) + 'd.txt')

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()

   # WORDS
    for i, word in enumerate(codes.word_index):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            w_embedding_matrix[i] = embedding_vector

    inptW = Input(shape=(max_len,))  # word input layer & embeddings
    embW = Embedding(input_dim=n_words, output_dim=w_embedding_dim,
                     input_length=max_len, weights=[w_embedding_matrix],
                     mask_zero=True)(inptW)

   # WORDS LC
    for i, wordLC in enumerate(codes.wordLC_index):
        embedding_vector = embeddings_index.get(wordLC)
        if embedding_vector is not None:
            wLC_embedding_matrix[i] = embedding_vector

    inptWLC = Input(shape=(max_len,))  # word input layer & embeddings
    embWLC = Embedding(input_dim=n_wordsLC, output_dim=wLC_embedding_dim,
                       input_length=max_len, weights=[wLC_embedding_matrix],
                       mask_zero=True)(inptWLC)

    inptS = Input(shape=(max_len,))  # suf input layer & embeddings
    embS = Embedding(input_dim=n_sufs, output_dim=50,
                     input_length=max_len, mask_zero=True)(inptS)

    inptP = Input(shape=(max_len,))  # suf input layer & embeddings
    embP = Embedding(input_dim=n_prefs, output_dim=50,
                     input_length=max_len, mask_zero=True)(inptP)

    inptExternalKnowledge = Input(shape=(max_len, 10))

    dropW = Dropout(0.3)(embW)
    dropWLC = Dropout(0.3)(embWLC)
    dropS = Dropout(0.3)(embS)
    dropP = Dropout(0.3)(embP)
    dropsExternalKnowledge = Dropout(0.0)(inptExternalKnowledge)
    drops = concatenate([dropW, dropWLC, dropS, dropP, dropsExternalKnowledge])

    # biLSTM
    bilstm = Bidirectional(LSTM(units=200, return_sequences=True,
                                recurrent_dropout=0.3))(drops)
    bilstm2 = Bidirectional(LSTM(units=200, return_sequences=True,
                                 recurrent_dropout=0.3))(bilstm)

    # Classifier
    model_top = tf.keras.models.Sequential()
    model_top.add(Dropout(0.3))
    model_top.add(Dense(200, activation='relu'))
    model_top.add(Dropout(0.3))
    model_top.add(Dense(50, activation='relu'))
    model_top.add(Dense(n_labels, activation='softmax'))

    out = TimeDistributed(model_top)(bilstm2)

    # build and compile model
    model = Model([inptW, inptWLC, inptS, inptP, inptExternalKnowledge], out)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model


# --------- MAIN PROGRAM -----------
# --
# -- Usage:  train.py ../data/Train ../data/Devel  modelname
# --
# directory with files to process
traindir = sys.argv[1]
validationdir = sys.argv[2]
modelname = sys.argv[3]

# load train and validation data
traindata = Dataset(traindir)
valdata = Dataset(validationdir)

# create indexes from training data
max_len = 150
suf_len = 5
pref_len = 5
codes = Codemaps(traindata, max_len, suf_len, pref_len)

# build network
model = build_network(codes)
with redirect_stdout(sys.stderr):
    model.summary()

# encode datasets
Xt = codes.encode_words(traindata)
Yt = codes.encode_labels(traindata)
Xv = codes.encode_words(valdata)
Yv = codes.encode_labels(valdata)

# train model

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    mode="min",
)

with redirect_stdout(sys.stderr):
    model.fit(Xt, Yt, batch_size=32, epochs=20, validation_data=(
        Xv, Yv), verbose=1, callbacks=[early_stopping])

# save model and indexs
model.save(modelname)
codes.save(modelname)
