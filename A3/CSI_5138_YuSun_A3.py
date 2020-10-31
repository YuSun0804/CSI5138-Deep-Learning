from operator import length_hint
from tensorflow.keras.layers import Embedding, AveragePooling1D
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, RNN,SimpleRNN,GlobalAveragePooling1D

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import models
from nltk.corpus import stopwords
import re
import string
import datetime
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import json
from tensorflow.python.keras.callbacks import Callback

print(tf.__version__)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

batch_size = 64
embedding_dim = 50
max_tokens=100000
sequence_length=300

path_to_glove_file = os.path.join(
    os.path.expanduser("/home/kevin/repo/python/CSI5138/A3"), "glove.6B/glove.6B.50d.txt"
)
path_to_train_file = os.path.join(
    os.path.expanduser("/home/kevin/repo/python/CSI5138/A3"), "aclImdb/train"
)
path_to_test_file = os.path.join(
    os.path.expanduser("/home/kevin/repo/python/CSI5138/A3"), "aclImdb/test"
)

from tensorflow.keras.preprocessing import text_dataset_from_directory

def loadnew():
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []
    for label_type in ['pos','neg']:
        dir_name = os.path.join(path_to_train_file,label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name,fname))
                text = f.read()
                text = preprocess_text(text)
                train_texts.append(text)
                f.close()
                if label_type == 'neg':
                    train_labels.append(0.0)
                else:
                    train_labels.append(1.0)
    for label_type in ['pos','neg']:
        dir_name = os.path.join(path_to_test_file,label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name,fname))
                text = f.read()
                text = preprocess_text(text)
                test_texts.append(text)
                f.close()
                if label_type == 'neg':
                    test_labels.append(0.0)
                else:
                    test_labels.append(1.0)
    return train_texts, test_texts, train_labels, test_labels

def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)
    
def loadData():
    raw_train_ds = text_dataset_from_directory(path_to_train_file, batch_size=batch_size)
    raw_test_ds = text_dataset_from_directory(path_to_test_file, batch_size=batch_size)
    print("train_size = {0} test_size = {1}".format(len(list(raw_train_ds)), len(list(raw_test_ds))))

    length_min = 1000
    length_max = 0
    all_length=[]
    for element in raw_train_ds.as_numpy_iterator():
        sentences = element[0]
        for s in sentences:
            l = len(s.decode("utf-8").split(" "))
            if l < length_min:
                length_min = l
            if l > length_max:
                length_max = l
            all_length.append(l)
    print("length_min = {0} length_max = {1}".format(length_min, length_max))
    
    plt.figure()
    plt.hist(all_length, bins=500)
    plt.xlim(left=0)
    plt.title("Histogram of sequance length")
    plt.xlabel("length bins")
    plt.ylabel("number of words")
    
    plt.figure()
    plt.title("Boxplot of sequance length")
    plt.boxplot(all_length, labels=['number of words'])

    return raw_train_ds, raw_test_ds


def custom_standardization(input_data):    
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    string_punc = tf.strings.regex_replace(stripped_html, "[%s]" % re.escape(string.punctuation), "")
    # for word in stop_words:
    #     s = "(\\s?\\b"+word+"\\b\\s?)"
    #     string_punc=tf.strings.regex_replace(string_punc, s, " ")
    return string_punc

def tokenizer(train_texts) :
    tokenizer = Tokenizer(num_words=max_tokens)
    tokenizer.fit_on_texts(train_texts)
    return tokenizer

def process(train_texts, test_texts, train_labels, test_labels, tokenizer):
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    train_data = pad_sequences(train_sequences,maxlen=sequence_length)
    test_data = pad_sequences(test_sequences, maxlen=sequence_length)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    return train_data,train_labels,test_data,test_labels

def vectorizer(raw_train_ds):
    vectorizer = TextVectorization(
        standardize=custom_standardization, max_tokens=max_tokens, output_sequence_length=sequence_length)
    text_ds = raw_train_ds.map(lambda x, y: x)
    vectorizer.adapt(text_ds)
    np.savetxt('voc.out', vectorizer.get_vocabulary(), fmt='%s')
    return vectorizer

def getWordIndex(vectorizer):
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    return word_index

def EmbeddingLayer(word_index):
    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    num_tokens = max_tokens + 2
    hits = 0
    misses = 0

    # Prepare embedding matrix
    missing_words=[]
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        if i > max_tokens:
            break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            missing_words.append(word)
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    np.savetxt('missing_words.out', missing_words, fmt='%s')

    embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        input_length=sequence_length,
        trainable=False
    )
    return embedding_layer

def MyLSTM(isMeanPoolinng,stateDim,embedding):
    model = models.Sequential()
    model.add(embedding)
    if isMeanPoolinng:
        model.add(LSTM(units=stateDim, return_sequences=True))
        model.add(GlobalAveragePooling1D(data_format='channels_first'))
    else:
        model.add(LSTM(units=stateDim))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()
    return model

def MyRNN(isMeanPoolinng,stateDim,embedding):
    model = models.Sequential()
    model.add(embedding)
    if isMeanPoolinng:
        model.add(SimpleRNN(units=stateDim, return_sequences=True))
        model.add(GlobalAveragePooling1D(data_format='channels_first'))
    else:
        model.add(SimpleRNN(units=stateDim))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()
    return model


def vectorize_text(text, label,vectorizer):
    text = tf.expand_dims(text, -1)
    return vectorizer(text), label


class CustomCallback(keras.callbacks.Callback):
    def __init__(self,path):
        if os.path.exists(path):
            os.remove(path)
        self.f = open(path, "a")

    def on_epoch_end(self, epoch, logs=None):
        self.f.write(json.dumps(logs)+"\r")
        self.f.flush()

if __name__ == "__main__":
    # raw_train_ds, raw_test_ds = loadData()
    # vectorizer = vectorizer(raw_train_ds)
    # word_index = getWordIndex(vectorizer)
    # train_ds = raw_train_ds.map(lambda x, y: vectorize_text(x, y, vectorizer))
    # test_ds = raw_test_ds.map(lambda x, y: vectorize_text(x, y, vectorizer))
    
    train_texts, test_texts, train_labels, test_labels = loadnew()
    tokenizer = tokenizer(train_texts)
    word_index = tokenizer.word_index
    train_data,train_labels,test_data,test_labels = process(train_texts, test_texts, train_labels, test_labels, tokenizer)
    idx = np.random.RandomState(seed=42).permutation(len(train_data))
    train_data,train_labels = train_data[idx], train_labels[idx]

    epochs= 30

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True)

    embedding = EmbeddingLayer(word_index)
    states = [20 ,50, 100, 200, 500]
    for state in states:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model = MyRNN(False, state, embedding)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        fit = model.fit(train_data,train_labels,epochs=epochs, verbose=2, validation_data=(test_data,test_labels),batch_size=batch_size,callbacks=[CustomCallback("loss_rnn_"+str(state)+".out")])
        result = model.evaluate(test_data, test_labels)
        np.savetxt('result_rnn_'+str(state)+'.out',  np.array(result), fmt=' % .04f')

        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # model = MyRNN(True, state, embedding)
        # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # fit = model.fit(train_data,train_labels,epochs=epochs, validation_data=(test_data,test_labels),batch_size=batch_size,callbacks=[CustomCallback("loss_rnn_mean_"+str(state)+".out")])
        # result = model.evaluate(test_data, test_labels)
        # np.savetxt('result_rnn_mean'+str(state)+'.out',  np.array(result), fmt=' % .04f')

        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # model = MyLSTM(False, state, embedding)
        # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # fit = model.fit(train_data,train_labels,epochs=epochs, validation_data=(test_data,test_labels),batch_size=batch_size,callbacks=[CustomCallback("loss_lstm_"+str(state)+".out")])
        # result = model.evaluate(test_data, test_labels)
        # np.savetxt('result_lstm_'+str(state)+'.out',  np.array(result), fmt=' % .04f')

        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # model = MyLSTM(True, state, embedding)
        # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # fit = model.fit(train_data,train_labels,epochs=epochs, validation_data=(test_data,test_labels),batch_size=batch_size,callbacks=[CustomCallback("loss_lstm_mean_"+str(state)+".out")])
        # result = model.evaluate(test_data, test_labels)
        # np.savetxt('result_lstm_mean_'+str(state)+'.out',  np.array(result), fmt=' % .04f')

    plt.show()
