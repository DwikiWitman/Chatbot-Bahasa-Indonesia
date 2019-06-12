import numpy as np
import csv
import os
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from attention_decoder import AttentionDecoder
from keras import backend as K

file = "clean-sample-data.tsv"
folder = "OpenSubtitles2018"

# load a clean dataset
def load_clean_sample_data():
    #return load(open(filename, 'rb'))
    filepath = os.path.join(folder,file)
    with open(filepath, "r", encoding="utf8") as read:
        reader = csv.reader(read,delimiter="\t")
        dataset = []
        for row in reader:
            dataset.append(row) 
    read.close()     
    return dataset

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer(char_level=False)
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    global prediksi
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    prediksi = prediction
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
        
    return ' '.join(target)

# translate
def translate(model, tokenizer, sources):
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, all_tokenizer, source)
    return translation

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

# load datasets
dataset = load_clean_sample_data()
dataset = np.reshape(dataset, (-1,2))
dataset1 = dataset.reshape(-1,1)

# prepare tokenizer
all_tokenizer = create_tokenizer(dataset1[:,0])
all_vocab_size = len(all_tokenizer.word_index) + 1
all_length = max_length(dataset1[:, 0])

# load model
model = load_model('Model/MODEL_GRU_LSTM_ATTNDECODER.h5',
                   custom_objects={'AttentionDecoder': AttentionDecoder, 
                                   'precision':precision, 'recall':recall, 'f1':f1})


# Setting up the chat
while(True):
    q = (input(str("YOU: ")))
    if q == 'bye':
        break
    q = q.strip().split('\n')

    #we tokenize
    X = all_tokenizer.texts_to_sequences(q)
    X = pad_sequences(X, maxlen=all_length, padding='post')
        
    # find reply and print it out
    a = translate(model, all_tokenizer, X)
    #a = set(a)
    words = a.split()
    #print('ANSWER: %s' % (thing))
    print ('ANSWER: ' + " ".join(sorted(set(words), key=words.index)))
