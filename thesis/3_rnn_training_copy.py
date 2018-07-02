import pandas as pd
import numpy as np
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM,Dense,Embedding,RepeatVector,TimeDistributed,Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint

n_units=128

#load the dataset
def load_file(filename):
    return load(open(filename,'rb'))

#saving the clean dataset
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('saved: %s' % filename)

def word_for_index(integer,tokenizer):
    for word,index in tokenizer.word_index.items():
        if index==integer:
            print("returning word: ",word)
            return word
    return None


#tokenisation
def create_tokenizer(lines):
    tokenizer=Tokenizer(lower=True,filters=' ',split='\n')#,split='\n')
    tokenizer.fit_on_texts(lines)
    #print(tokenizer.word_index.items())
    return tokenizer

#max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

#encoding and padding sequences
def encoding_sequences(tokenizer,length,lines):
    x=tokenizer.texts_to_sequences(lines)
    x=pad_sequences(x,maxlen=length,padding='post')
    return x

#one hot vector of target sequence
def encode_output(sequences, vocab_size):
    ylist=list()
    for sequence in sequences:
        encoded=to_categorical(sequence,num_classes=vocab_size)
        #print(encoded)
        ylist.append(encoded)
    y=array(ylist)
    y=y.reshape(sequences.shape[0],sequences.shape[1],vocab_size)
    return y


def define_model(src,target,src_timesteps,tar_timesteps,n_units):
    model=Sequential()
    e = Embedding(src, 128, weights=[embedding_matrix], input_length=src_timesteps, trainable=False)
    model.add(e)
    #model.add(LSTM(units=n_units, return_sequences=False))
    #model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units,return_sequences=True))
    model.add(TimeDistributed(Dense(target, activation='softmax')))
    return model




#loading the datasets
dataset=load_file('14_subobj.pkl')
train=load_file("14_subobj_train.pkl")
test=load_file("14_subobj_test.pkl")

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


#preparing training data
trainX=encoding_sequences(sub_tokenizer,sub_length,train[:,0])
print("trainx shape:",trainX.shape)
trainY=encoding_sequences(obj_tokenizer,obj_length,train[:,1])
print("trainY shape: ",trainY.shape)
trainY=encode_output(trainY,obj_size)
print("target_trainY shape:",trainY.shape)



#preparing test data
testX=encoding_sequences(sub_tokenizer,sub_length,test[:,0])
print(testX.shape)
testY=encoding_sequences(obj_tokenizer,obj_length,test[:,1])
testY=encode_output(testY,obj_size)
print("target_trainY shape:",testY.shape)



df=pd.read_csv("15_allemb.emb",sep='\t')
df.columns=["word","embedding"]
df_word=df["word"]
word_index_dict=df_word.to_dict()
df_embed=df["embedding"]
df_embed.to_csv("15_coefs.csv",header=None,index=False,sep='\n')

with open("15_coefs.csv","r")as w:
    embedding_matrix=np.array([[float(digit) for digit in line.split()]for line in w])



#calling model
model=define_model(sub_size,obj_size,sub_length,obj_length,128)
model.compile(optimizer='adam',loss='binary_crossentropy')#loss='categorical_crossentropy'

#print summary
print(model.summary())
plot_model(model,to_file='model_14.png',show_shapes=True)
#fitting
filename='model_14.h5'
checkpoint=ModelCheckpoint(filename,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
model.fit(trainX,trainY,epochs=10,batch_size=64,validation_data=(testX,testY),callbacks=[checkpoint],verbose=2)