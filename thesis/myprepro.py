
# -*- coding: utf-8 -*-
#file to convert subjects and objects to lower case and save in .pkl files

import string
import regex as re
from pickle import dump
from numpy import array
import numpy as np
import pandas as pd

#load file into memory
def load_doc(filename):
    file=open(filename,mode='rt',encoding="utf-8")
    text=file.read()
    file.close()
    return text

def divideintopairs(doc):
    lines=doc.strip().split("\n")
    pairs=[line.split("\t") for line in lines]
    print(type(pairs))
    return pairs

#clean the list of sentences:
def clean_pairs(lines):
    cleaned=list()
    for pair in lines:
        clean_pair=list()
        for line in pair:
            line=line.split()
            #convert to lower case
            line=[word.lower() for word in line]
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)

#saving clean sentences to a file
def save_clean_data(sentences, filename):
    print(type(sentences))
    dump(sentences, open(filename, 'wb'))
    print('saved: %s' % filename)



#loading the dataset
filename="27testing.csv"
doc=load_doc(filename)
#splitting
pairs= divideintopairs(doc)
#cleaning
clean_pairs=clean_pairs(pairs)
#saving the cleaned data
save_clean_data(clean_pairs,"27testing.pkl")

for i in range(520,530):
    print("%s - %s"% (clean_pairs[i,0],clean_pairs[i,1]) )





