import csv
import os
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
from spacy.lang.id import Indonesian
import spacy

nlp_indonesia = Indonesian()  # use directly
nlp_indonesia = spacy.blank('id')

# procedure clean noise dataset
def preprocessing_text(text):
    text = text.lower()
    #text = re.sub('[^a-zA-Z0-9 .,?!]', '', text)
    text = re.sub(r'[^a-zA-Z\s.,?!]', u'', text, flags=re.UNICODE)
    
#    for r in (
#            (" ku", " aku"), (" gw", " aku"), (" saya", " aku"), (" gue", " aku"), (" gua", " aku"),
#            (" anda", " kamu"), (" lu", " kamu"), (" kau", " kamu"), (" mu", " kamu"),
#            (" dia", " dia"), (" doi", " dia"),
#            (" kita", " kami"),
#            (" tak", " tidak"), (" engga", " tidak"), (" enggak", " tidak"), (" ga", " tidak"), (" gak", " tidak"),
#            (" ya", " iya"), (" yes", " iya"), (" yoi", " iya"), (" yah", " iya"),
#            (" hei", " hai"), (" hey", " hai"), (" halo", " hai"), (" hay", " hai")
#    ):
#        text = text.replace(*r)
    
    aku = ['ku', 'gw', 'saya', 'gue', 'gua'] 
    kamu = ['anda', 'lu', 'kau', 'mu'] 
    dia = ['doi']
    kami = ['kita']
    tidak = ['tak', 'engga', 'enggak', 'ga', 'gak'] 
    iya = ['ya', 'yes', 'yoi', 'yah'] 
    hai = ['hei', 'hey', 'halo', 'hay'] 
    
    lines = []
    for word in text.split():
        if word in aku:
            lines.append("aku")
        elif word in kamu:
            lines.append("kamu")
        elif word in dia:
            lines.append("dia")
        elif word in kami:
            lines.append("kami")
        elif word in tidak:
            lines.append("tidak")
        elif word in iya:
            lines.append("iya")
        elif word in hai:
            lines.append("hai")
        else:
            lines.append(word)
            
    text = ' '.join(lines)  
#    text = re.sub('(ku|gw|saya|gue|gua)','aku',text)
#    text = re.sub('(anda|lu|kau|mu)','kamu',text)
#    text = re.sub('(dia|doi|nya)','dia',text)
#    text = re.sub('(kita)','kami',text)
#    text = re.sub('(tak|engga|enggak|ga|gak)','tidak',text)
#    text = re.sub('(ya|yes|yoi|yah)','iya',text)
#    text = re.sub('(hei|hey|halo|hay)','hai',text)
    text = ' '.join(text.split())
    #text = re.sub('([\w]+)([,;.?!#&\'\"-]+)([\w]+)?', r'\1 \2 \3', text)
    #text = re.sub(' +', ' ', text) 
    
    maxlen = 15
    if len(text.split()) > maxlen:
        text = (' ').join(text.split()[:maxlen])
        
#    text = nlp_indonesia(text)
#    text = [token.text for token in text]
##    text = word_tokenize(text)
#    text = TreebankWordDetokenizer().detokenize(text)
#    text = " ".join(text.split())
#    text = re.sub('[.]', ' .', text)
#    text = re.sub('[,]', ' ,', text)
#    text = re.sub('[?]', ' ?', text)
#    text = re.sub('[!]', ' !', text)
    return text

# read OpenSubtitles2018
lines_filepath = os.path.join("OpenSubtitles2018","OpenSubtitles2018.tokenized.id")
with open(lines_filepath, "r", encoding="utf8") as lines:
    array = []
    for line in lines:
        line = preprocessing_text(line.rstrip('\n'))
        array.append(line)
lines.close()
# write context-target
lines_filepath = os.path.join("OpenSubtitles2018","context-target-coba.tsv")
with open(lines_filepath, "w", encoding="utf-8",newline='') as lines:
    writer = csv.writer(lines, delimiter='\t')
    i = 0
    while i < len(array):
        try:
            writer.writerow([array[i], array[i+1]])
        except:
            pass
        i+=1      
lines.close()
#del array,i,line,lines_filepath

#import pickle
#from numpy.random import shuffle
#from numpy import loadtxt
## load a clean dataset
#def load_clean_sentences(filename):
#	return pickle.load(open(filename, 'rb'))
## save a list of clean sentences to file
#def save_clean_data(sentences, filename):
#	pickle.dump(sentences, open(filename, 'wb'), pickle.HIGHEST_PROTOCOL)
#	print('Saved: %s' % filename)
## load dataset
#a = loadtxt('OpenSubtitles2018/context-target.id', delimiter='|', dtype=str, encoding="utf8")
## reduce dataset size
#n_sentences = 2000000
#dataset = a[:n_sentences, :]
## random shuffle
#shuffle(dataset)
## split into train/test
#train, test = dataset[:int(n_sentences*70/100)], dataset[int(n_sentences*70/100):]
## save
#save_clean_data(dataset, 'OpenSubtitles2018/both.pkl')
#save_clean_data(train, 'OpenSubtitles2018/train.pkl')
    #save_clean_data(test, 'OpenSubtitles2018/test.pkl')