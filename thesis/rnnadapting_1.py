import pandas as pd
import numpy as np
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM,Dense,Embedding,RepeatVector,TimeDistributed,Input,Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from numpy import zeros
import tensorflow as tf
import matplotlib.pyplot as plt2
plt2.switch_backend('agg')
import matplotlib.pyplot as plt1
plt1.switch_backend('agg')


n_units=128



#load the dataset
def load_clean_sentences(filename):
    return load(open(filename,'rb'))

#saving the clean dataset
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('saved: %s' % filename)

def word_for_id(integer,tokenizer):
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
def encode_sequences(tokenizer,length,lines):
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
    e = Embedding(src, 128, weights=[em], input_length=src_timesteps, trainable=False,mask_zero=True)
    model.add(e)
    #model.add(LSTM(units=n_units,return_sequences=True))
    model.add(LSTM(units=n_units, return_sequences=False))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(n_units,return_sequences=True))
    model.add(LSTM(n_units,return_sequences=True))
    model.add(TimeDistributed(Dense(target, activation='softmax')))
    return model



#loading the datasets
#enter the output test and train .pkl filenames
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


#preparing training data
trainX=encode_sequences(sub_tokenizer,sub_length,train[:,0])
print("trainx shape:",trainX.shape)
trainY=encode_sequences(obj_tokenizer,obj_length,train[:,1])
print("trainY shape: ",trainY.shape)


trainY=encode_output(trainY,obj_size) #sub_size
print("target_trainY shape:",trainY.shape)

#preparing test data
testX=encode_sequences(sub_tokenizer,sub_length,test[:,0])
print("testX shape:",testX.shape)

testY=encode_sequences(obj_tokenizer,obj_length,test[:,1])

testY=encode_output(testY,obj_size)
print("testY shape:",testY.shape)


df=pd.read_csv("27testing_80wl_10nm_online.emb",sep='\t')
df.columns=["word","embedding"]
df_word=df["word"]
word_index_dict=df_word.to_dict()
df_embed=df["embedding"]


em=zeros((sub_size,n_units))

for word,i in sub_tokenizer.word_index.items():
        #index=word_index_dict.get(word)
        x=df.index[df["word"].str.match(word)]
        #print(word,i,x[0])
        em[i]=np.array([[float(digit) for digit in df.at[x[0],"embedding"].split()]])



#calling model
model=define_model(sub_size,obj_size,sub_length,obj_length,128)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#print summary
print(model.summary())
plot_model(model,to_file='model_27testing_100epochs_d50_b128_L3.png',show_shapes=True)
#fitting
filename='model_27testing_60epochs_d50_b128_L3.h5'
checkpoint=ModelCheckpoint(filename,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
history=model.fit(trainX,trainY,epochs=60,batch_size=128,validation_data=(testX,testY),callbacks=[checkpoint],verbose=2)

# Plot training & validation accuracy values
plt1.plot(history.history['acc'])
plt1.plot(history.history['val_acc'])
plt1.title('Model accuracy')
plt1.ylabel('Accuracy')
plt1.xlabel('Epoch')
plt1.legend(['Train', 'Test'], loc='upper left')
plt1.savefig("27testing_100epochs_d50_b128_L3_acc.png")
plt1.close()


# Plot training & validation loss values
plt2.plot(history.history['loss'])
plt2.plot(history.history['val_loss'])
plt2.title('Model loss')
plt2.ylabel('Loss')
plt2.xlabel('Epoch')
plt2.legend(['Train', 'Test'], loc='upper left')
plt2.savefig("27testing_100epochs_d50_b128_L3_loss.png")
