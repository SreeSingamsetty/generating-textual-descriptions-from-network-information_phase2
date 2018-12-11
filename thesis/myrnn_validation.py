# -*- coding: utf-8 -*-
#file for inference phase
from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

#enter the desired result set filenames

trainresultfilename="27testing_100epochs_d50_b128_L3_train.txt"
testresultfilename="27testing_100epochs_d50_b128_L3_test.txt"
#load the dataset
def load_clean_sentences(filename):
    return load(open(filename,'rb'))

#tokenisation
def create_tokenizer(lines):
    tokenizer=Tokenizer(lower=True,filters=' ')
    tokenizer.fit_on_texts(lines)
    return tokenizer

#max length of subjects and objects
def max_length(lines):
    return max(len(line.split()) for line in lines)

#encoding and padding sequences
def encode_sequences(tokenizer, length,lines):
    x=tokenizer.texts_to_sequences(lines)
    x=pad_sequences(x,maxlen=length,padding='post')
    return x

#mapping integer to a word
def word_for_id(integer,tokenizer):
    for word,index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

#generating object
def predict_obj(model,tokenizer,sub):
    prediction=model.predict(sub,verbose=0)[0]
    integers=[argmax(vector)for vector in prediction]
    obj=list()
    for i in integers:
        word=word_for_id(i,tokenizer)
        if word is None:
            break
        obj.append(word)
    return ' '.join(obj)





#evaluation
def evaluate_model(model,tokenizer,subjects,raw_dataset,filename):
    actual,predicted=list(),list()

    with open(filename, "w") as wr:
        for i,source in enumerate(subjects):
            source=source.reshape((1,source.shape[0]))
            translation=predict_obj(model,obj_tokenizer,source)
            raw_sub,raw_obj=raw_dataset[i]
            actual.append(raw_obj.split())
            predicted.append(translation.split())

            wr.write(raw_sub)
            wr.write("\t")
            wr.write(translation)
            wr.write("\n")



#loading the datasets
dataset=load_clean_sentences('27testing.pkl')
train=load_clean_sentences("27testing_train.pkl")
test=load_clean_sentences("27testing_test.pkl")

#subj tokenizer
sub_tokenizer=create_tokenizer(dataset[:,0])
sub_size=len(sub_tokenizer.word_index)+1
sub_length=max_length(dataset[:,0])
print("subject vocabulary size:%d" % sub_size)
print("maximum length of a subject:%d" % (sub_length))

#obj tokenizer
obj_tokenizer = create_tokenizer(dataset[:, 1])
obj_size = len(obj_tokenizer.word_index) + 1
obj_length = max_length(dataset[:, 1])
print("object vocabulary size:%d" % obj_size)
print("maximum length of a object:%d" % (obj_length))

#preparing subjects from train and test sets
trainx=encode_sequences(sub_tokenizer,sub_length,train[:,0])
print(trainx.shape)
testx=encode_sequences(sub_tokenizer,sub_length,test[:,0])
print(testx.shape)


#loading the model:
model=load_model('model_27testing_60epochs_d50_b128_L3.h5')
print('training in progress')
evaluate_model(model,obj_tokenizer,trainx,train,trainresultfilename)

print("testing in progress")
evaluate_model(model,obj_tokenizer,testx,test,testresultfilename)
