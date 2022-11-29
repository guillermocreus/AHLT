
import string
import re

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

from string import punctuation

from dataset import *

class Codemaps :
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, maxlen=None, suflen=None, preflen=None) :
        
        self.pos_index = {'UNK': 0, 'PAD': -1, 'n': 1, 'v': 2, 'a': 3, 'r': 4}
        self.external_index = {'UNK': 0, 'PAD': -1, 'drug': 1, 'group': 2, 
                               'brand': 3, 'drug_n': 4}
        self.external = {}
        
        with open("resources/HSDB.txt") as h:
            for x in h.readlines():
                self.external[x.strip().lower()] = self.external_index["drug"]
        with open("resources/drug_n.txt") as h:
            for x in h.readlines():
                self.external[x.strip().lower()] = self.external_index["drug_n"]
        with open("resources/DrugBank.txt") as h:
            for x in h.readlines():
                (n, t) = x.strip().lower().split("|")
                self.external[n] = self.external_index[t]

        if isinstance(data, Dataset) and maxlen is not None and suflen is not None:
            self.__create_indexs(data, maxlen, suflen, preflen)
            
        elif type(data) == str and maxlen is None and suflen is None and preflen is None:
            self.__load(data)
            
        else:
            print('codemaps: Invalid or missing parameters in constructor')
            exit()

            
    # --------- Create indexs from training data
    # Extract all words and labels in given sentences and 
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen, preflen) :

        self.maxlen = maxlen
        self.suflen = suflen
        self.preflen = preflen
        words = set([])
        wordsLC = set([])
        lc_words = set([])
        sufs = set([])
        prefs = set([])
        labels = set([])
        
        for s in data.sentences() :
            for t in s :
                words.add(t['form'])
                wordsLC.add(t['lc_form'])
                sufs.add(t['lc_form'][-self.suflen:])
                prefs.add(t['lc_form'][-self.preflen:])
                labels.add(t['tag'])

        self.word_index = {w: i+2 for i,w in enumerate(list(words))}
        self.word_index['PAD'] = 0 # Padding
        self.word_index['UNK'] = 1 # Unknown words
        
        self.wordLC_index = {w: i+2 for i,w in enumerate(list(wordsLC))}
        self.wordLC_index['PAD'] = 0 # Padding
        self.wordLC_index['UNK'] = 1 # Unknown words

        self.suf_index = {s: i+2 for i,s in enumerate(list(sufs))}
        self.suf_index['PAD'] = 0  # Padding
        self.suf_index['UNK'] = 1  # Unknown suffixes

        self.pref_index = {s: i+2 for i,s in enumerate(list(prefs))}
        self.pref_index['PAD'] = 0  # Padding
        self.pref_index['UNK'] = 1  # Unknown prefixes

        self.label_index = {t: i+1 for i,t in enumerate(list(labels))}
        self.label_index['PAD'] = 0 # Padding
        
    ## --------- load indexs ----------- 
    def __load(self, name) : 
        self.maxlen = 0
        self.suflen = 0
        self.preflen = 0
        self.word_index = {}
        self.wordLC_index = {}
        self.suf_index = {}
        self.pref_index = {}
        self.label_index = {}

        with open(name+".idx") as f :
            for line in f.readlines(): 
                (t,k,i) = line.split()
                if t == 'MAXLEN' : self.maxlen = int(k)
                elif t == 'SUFLEN' : self.suflen = int(k)                
                elif t == 'PREFLEN' : self.preflen = int(k)                
                elif t == 'WORD': self.word_index[k] = int(i)
                elif t == 'WORDLC': self.wordLC_index[k] = int(i)
                elif t == 'SUF': self.suf_index[k] = int(i)
                elif t == 'PREF': self.pref_index[k] = int(i)
                elif t == 'LABEL': self.label_index[k] = int(i)
                            
    
    ## ---------- Save model and indexs ---------------
    def save(self, name) :
        # save indexes
        with open(name+".idx","w") as f :
            print ('MAXLEN', self.maxlen, "-", file=f)
            print ('SUFLEN', self.suflen, "-", file=f)
            print ('PREFLEN', self.preflen, "-", file=f)
            for key in self.label_index : print('LABEL', key, self.label_index[key], file=f)
            for key in self.word_index : print('WORD', key, self.word_index[key], file=f)
            for key in self.wordLC_index : print('WORDLC', key, self.wordLC_index[key], file=f)
            for key in self.suf_index : print('SUF', key, self.suf_index[key], file=f)
            for key in self.pref_index : print('PREF', key, self.pref_index[key], file=f)


    ## --------- encode X from given data ----------- 
    def encode_words(self, data) :        
        # encode and pad sentence words
        Xw = [[self.word_index[w['form']] if w['form'] in self.word_index else self.word_index['UNK'] for w in s] for s in data.sentences()]
        Xw = pad_sequences(maxlen=self.maxlen, sequences=Xw, padding="post", value=self.word_index['PAD'])
        # encode and pad sentence words LC
        XwLC = [[self.wordLC_index[w['lc_form']] if w['lc_form'] in self.wordLC_index else self.wordLC_index['UNK'] for w in s] for s in data.sentences()]
        XwLC = pad_sequences(maxlen=self.maxlen, sequences=XwLC, padding="post", value=self.wordLC_index['PAD'])
        # encode and pad suffixes
        Xs = [[self.suf_index[w['lc_form'][-self.suflen:]] if w['lc_form'][-self.suflen:] in self.suf_index else self.suf_index['UNK'] for w in s] for s in data.sentences()]
        Xs = pad_sequences(maxlen=self.maxlen, sequences=Xs, padding="post", value=self.suf_index['PAD'])
        # encode and pad prefixes
        Xp = [[self.pref_index[w['lc_form'][-self.preflen:]] if w['lc_form'][-self.preflen:] in self.pref_index else self.pref_index['UNK'] for w in s] for s in data.sentences()]
        Xp = pad_sequences(maxlen=self.maxlen, sequences=Xp, padding="post", value=self.pref_index['PAD'])
        
        # Xext = [[self.external_index[self.external[w['form']]] if w['form'] in self.external else self.external_index['UNK'] for w in s] for s in data.sentences()]
        # Xext = pad_sequences(maxlen=self.maxlen, sequences=Xext, padding="post", value=self.external_index['PAD'])   
        
        most_common_suffixes = {
            3: {
                'drug': ['ide', 'cin', 'ole', 'one'],
                'brand': ['rin', 'CIN', 'XOL', 'SYS', 'RON'],
                'group': ['nts', 'ics', 'nes', 'ors', 'ids'],
                'drug_n': ['ate', 'PCP', 'ANM', '-MC']
            },

            4: {
                'drug': ['pine', 'zole', 'mine', 'arin'],
                'brand': ['irin', 'OCIN', 'AXOL', 'ASYS', 'IOXX'],
                'group': ['tics', 'ants', 'tors', 'ines', 'ents'],
                'drug_n': ['NANM', 'aine', '8-MC']
            },

            5: {
                'drug': ['azole', 'amine', 'farin', 'mycin'],
                'brand': ['pirin', 'DOCIN', 'TAXOL', 'GASYS', 'VIOXX'],
                'group': ['gents', 'itors', 'sants', 'etics', 'otics'],
                'drug_n': ['gaine', '18-MC', '-NANM']
            }
        }
        
        Xext = []
        for s in data.sentences():
            sent_feat = []
            for w in s:
                w_feat = []
                
                # POS tag
                if self.pos_index.get(w['pos']) is not None:
                    w_feat.append(self.pos_index.get(w['pos']))
                else:
                    w_feat.append(self.pos_index['UNK'])
                
                # External knowledge
                if self.external_index.get(w['form']) is not None:
                    w_feat.append(self.external_index.get(w['form']))
                else:
                    w_feat.append(self.external_index['UNK'])
                    
                # Dashes    
                n_dashes = len(re.findall('-', w['form']))
                if n_dashes:
                    w_feat.append(n_dashes)
                else:
                    w_feat.append(self.external_index['UNK'])
                # No. of uppercase
                n_upper = sum(i.isupper() for i in w['form'])
                if n_upper:
                    w_feat.append(n_upper)
                else:
                    w_feat.append(self.external_index['UNK'])
                # No. of digits
                n_digits = len(re.findall('\d', w['form']))
                if n_digits:
                    w_feat.append(n_digits)
                else:
                    w_feat.append(self.external_index['UNK'])
                # 3, 4, 5 suffixes
                for n_suffix in [3, 4, 5]:
                    suffix_class = self.external_index['UNK']
                    for d_class in ['drug', 'brand', 'group', 'drug_n']:
                        if w['form'][-n_suffix:] in most_common_suffixes[n_suffix][d_class]:
                            suffix_class = self.external_index[d_class]
                            
                    w_feat.append(suffix_class)
                # Punctuation
                if w['form'] in punctuation:
                    w_feat.append(1)
                else:
                    w_feat.append(self.external_index['UNK'])
                # Slashes    
                if '/' in w['form']:
                    w_feat.append(1)
                else:
                    w_feat.append(self.external_index['UNK'])

                sent_feat.append(w_feat)
            
            Xext.append(sent_feat)
            
        Xext = pad_sequences(maxlen=self.maxlen, sequences=Xext, padding="post", value=self.external_index['PAD'])        
    
        # return encoded sequences
        return [Xw, XwLC, Xs, Xp, Xext]

    
    ## --------- encode Y from given data ----------- 
    def encode_labels(self, data) :
        # encode and pad sentence labels 
        Y = [[self.label_index[w['tag']] for w in s] for s in data.sentences()]
        Y = pad_sequences(maxlen=self.maxlen, sequences=Y, padding="post", value=self.label_index["PAD"])
        return np.array(Y)

    ## -------- get word index size ---------
    def get_n_words(self) :
        return len(self.word_index)
    ## -------- get word index size ---------
    def get_n_wordsLC(self) :
        return len(self.wordLC_index)
    ## -------- get suf index size ---------
    def get_n_sufs(self) :
        return len(self.suf_index)
    ## -------- get suf index size ---------
    def get_n_prefs(self) :
        return len(self.pref_index)
    ## -------- get label index size ---------
    def get_n_labels(self) :
        return len(self.label_index)
    
    def get_n_external(self):
        return len(self.external_index)

    ## -------- get index for given word ---------
    def word2idx(self, w) :
        return self.word_index[w]
    ## -------- get index for given suffix --------
    def suff2idx(self, s) :
        return self.suff_index[s]
    ## -------- get index for given prefix --------
    def pref2idx(self, s) :
        return self.pref_index[s]
    ## -------- get index for given label --------
    def label2idx(self, l) :
        return self.label_index[l]
    ## -------- get label name for given index --------
    def idx2label(self, i) :
        for l in self.label_index :
            if self.label_index[l] == i:
                return l
        raise KeyError

