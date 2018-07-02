from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle
import numpy as np

#loading the clean dataset
def load_clean_sentences(filename):
    return load(open(filename,'rb'))

#saving the clean dataset
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('saved: %s' % filename)

#load dataset
dataset=load_clean_sentences("16_subobj.pkl")
#shuffling the dataset
shuffle(dataset)
#split into train and test
train,test=dataset[:5000],dataset[5000:]
print(train[0:10])
print(test[0:10])
#saving
save_clean_data(dataset,'16_subobj.pkl')
save_clean_data(train,'16_subobj_train.pkl')
save_clean_data(test,'16_subobj_test.pkl')
