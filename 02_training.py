import csv
import os
import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
#from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dense , Dropout, Activation, Permute, Multiply,SimpleRNN
from keras.layers.normalization import BatchNormalization
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.optimizers import Adam
#from keras.layers import Bidirectional
#from keras import regularizers
#from keras.callbacks import ModelCheckpoint
#import tensorflow as tf
from keras.callbacks import TensorBoard
from attention_decoder import AttentionDecoder
from keras import backend as K

file = "sample-data.tsv"
folder = "OpenSubtitles2018"
# load a clean dataset
def load_clean_sample_data(folder, file):
    #return load(open(filename, 'rb'))
    lines_filepath = os.path.join(folder,file)
    with open(lines_filepath, "r", encoding="utf8") as read:
        reader = csv.reader(read,delimiter="\t")
        dataset = []
        for row in reader:
            dataset.append(row) 
    read.close()     
    return dataset

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # intereply encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    
def f1(y_true, y_pred):
    result_precision = precision(y_true, y_pred)
    result_recall = recall(y_true, y_pred)
    return 2*((result_precision*result_recall)/(result_precision+result_recall+K.epsilon()))

# load embedding as a dict
def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename,'rb')
    lines = file.readlines()[1:]
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
    return embedding

def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, 100))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = embedding.get(word)
    return weight_matrix

# define NMT model
def define_model(vocab, timesteps, n_units, encoder, decoder, attention):
    model = Sequential()
    model.add(Embedding(vocab, n_units, input_length=timesteps, mask_zero=True))
    #model.add(Embedding(vocab, n_units, weights=[embedding_vectors], input_length=timesteps, trainable=False))
    if(encoder == "LSTM"):
        model.add(LSTM(n_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
    elif(encoder == "GRU"):
        model.add(GRU(n_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
    
    model.add(RepeatVector(timesteps))
    if(decoder == "LSTM"):
        model.add(LSTM(n_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    elif(decoder == "GRU"):
        model.add(GRU(n_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    
    model.add(BatchNormalization())
    if(attention == "ATTNDECODER"):
        model.add(AttentionDecoder(n_units, vocab))
    else:
        model.add(TimeDistributed(Dense(vocab, activation='softmax',
                                        #kernel_regularizer=regularizers.l2(0.01),
                                        #activity_regularizer=regularizers.l2(0.01)
                                        )))
    return model

# define NMT model
def define_model_rnn(vocab, timesteps, n_units, encoder, decoder, attention):
    model = Sequential()
    model.add(Embedding(vocab, n_units, input_length=timesteps, mask_zero=True))
    #model.add(Embedding(vocab, n_units, weights=[embedding_vectors], input_length=timesteps, trainable=False))
    model.add(SimpleRNN(n_units, return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(SimpleRNN(n_units, return_sequences=True))
    #model.add(BatchNormalization())
    if(attention == "ATTNDECODER"):
        model.add(AttentionDecoder(n_units, vocab))
    else:
        model.add(TimeDistributed(Dense(vocab, activation='tanh',
                                        #kernel_regularizer=regularizers.l2(0.01),
                                        #activity_regularizer=regularizers.l2(0.01)
                                        )))
    return model

def define_model_lstm_lstm(vocab, timesteps, n_units):
    model = Sequential()
    model.add(Embedding(vocab, n_units, input_length=timesteps, mask_zero=True))
    model.add(LSTM(n_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(n_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(BatchNormalization())
    model.add(AttentionDecoder(n_units, vocab))
    return model


def define_model_lstm_gru(vocab, timesteps, n_units):
    model = Sequential()
    model.add(Embedding(vocab, n_units, input_length=timesteps, mask_zero=True))
    model.add(LSTM(n_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
    model.add(RepeatVector(timesteps))
    model.add(GRU(n_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(BatchNormalization())
    model.add(AttentionDecoder(n_units, vocab))
    return model

def define_model_gru_gru(vocab, timesteps, n_units):
    model = Sequential()
    model.add(Embedding(vocab, n_units, input_length=timesteps, mask_zero=True))
    model.add(GRU(n_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
    model.add(RepeatVector(timesteps))
    model.add(GRU(n_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(BatchNormalization())
    model.add(AttentionDecoder(n_units, vocab))
    return model

def define_model_gru_lstm(vocab, timesteps, n_units):
    model = Sequential()
    model.add(Embedding(vocab, n_units, input_length=timesteps, mask_zero=True))
    model.add(GRU(n_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(n_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(BatchNormalization())
    model.add(AttentionDecoder(n_units, vocab))
    return model
#import pandas as pd
#import h5py
#from sklearn.model_selection import train_test_split
#df = pd.read_csv("OpenSubtitles2018/context-target.id 2",sep='|', lineterminator='\r') 
#df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
#df_temp = df.values.reshape(-1,1)
#df = pd.DataFrame(df_temp)
#del df_temp

## prepare tokenizer
#all_tokenizer = create_tokenizer(df.values.astype(str)[:, 0])
#all_vocab_size = len(all_tokenizer.word_index) + 1
#all_length = max_length(df.values.astype(str)[:, 0])
#print('ALL Vocabulary Size: %d' % (all_vocab_size))
#print('ALL Max question length: %d' % (all_length))
#del df

## prepare training data
#trainX = encode_sequences(all_tokenizer, all_length, df_train.values.astype(str)[:, 0])
##save array as hdf5
#with h5py.File('trainX.h5', 'w') as hf:
#    hf.create_dataset("trainX",  data=trainX)
##df_trainX = pd.DataFrame(data=trainX[0:,0:])
#del trainX
##df_trainX.to_pickle('df_trainX.pkl')
##del df_trainX
#
#trainY = encode_sequences(all_tokenizer, all_length, df_train.values.astype(str)[:, 1])
#del df_train
##df_trainY = pd.DataFrame(data=trainY[0:,0:])
#with h5py.File('trainY.h5', 'w') as hf:
#    hf.create_dataset("trainY",  data=trainY)
##h5f = h5py.File('dataset.h5','r')
##h5f['dataset_1'][:]
#
#ylist = list()
##ylist = []
#df_temp = pd.Series()
#for sequence in trainY:
#    encoded = to_categorical(sequence, num_classes=all_vocab_size)
#    ylist.append(encoded)
#    #df_temp.append(pd.DataFrame(encoded))
##y = array(ylist)
#y = df_temp.values.astype(array)
#y = y.reshape(trainY.shape[0], trainY.shape[1], all_vocab_size)
#    
##trainY = encode_output(trainY, all_vocab_size)



#load datasets
dataset = load_clean_sample_data("OpenSubtitles2018","clean-sample-data.tsv")
dataset = np.reshape(dataset, (-1,2))
dataset1 = dataset.reshape(-1,1)
train, test = dataset[ : int(len(dataset)*80/100) ],  dataset[ int(len(dataset)*80/100): ]
del dataset

# prepare tokenizer
all_tokenizer = create_tokenizer(dataset1[:, 0])
print(all_tokenizer)
all_vocab_size = len(all_tokenizer.word_index) + 1
all_length = max_length(dataset1[:, 0])
print('ALL Vocabulary Size: %d' % (all_vocab_size))
print('ALL Max question length: %d' % (all_length))
del dataset1

# prepare training data
trainX = encode_sequences(all_tokenizer, all_length, train[:, 0])
trainY = encode_sequences(all_tokenizer, all_length, train[:, 1])
trainY = encode_output(trainY, all_vocab_size)
del train

# prepare validation data
testX = encode_sequences(all_tokenizer, all_length, test[:, 0])
testY = encode_sequences(all_tokenizer, all_length, test[:, 1])
testY = encode_output(testY, all_vocab_size)
del test

# define model (yg bagian ini yg diedit sesuai kemauan dgn menyesuaikan pemanggilan function2 diatas)
#raw_embedding = load_embedding('Word2vec/idwiki_word2vec.txt')
#embedding_vectors = get_weight_matrix(raw_embedding, all_tokenizer.word_index)
#model = define_model_embed(all_vocab_size, all_length, 100, encoder, decoder, embedding_vectors)
encoder = "LSTM"
decoder = "LSTM"
attention = "ATTNDECODER"
model = define_model_rnn(all_vocab_size, all_length, 256, encoder, decoder, attention)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy', precision, recall, f1])
#model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics = ['accuracy', precision, recall, f1])

# summarize defined model
print(model.summary())
#tensorboard --logdir ./Graph
#tensorboard --logdir=LSTM-LSTM:./Model/15rb/lstm-lstm,LSTM-GRU:./Model/15rb/lstm-gru,GRU-GRU:./Model/15rb/gru-gru,GRU-LSTM:./Model/15rb/gru-lstm
tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
#train and save model
import timeit
start_time = timeit.default_timer()
history = model.fit(trainX, trainY, epochs=400, batch_size=64, validation_data=(testX, testY), verbose=1,callbacks=[tensor_board])
print(timeit.default_timer() - start_time)

import timeit
start_time = timeit.default_timer()
score = model.evaluate(testX, testY, batch_size=64)
print(timeit.default_timer() - start_time)
print(score)

import datetime
str(datetime.timedelta(seconds=3437))

filename = 'Model/MODEL_'+encoder+'_'+decoder+'_'+attention+'.h5'
model.save(filename)

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
