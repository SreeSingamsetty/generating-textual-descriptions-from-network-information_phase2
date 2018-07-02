from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu

#load the dataset
def load_file(filename):
    return load(open(filename,'rb'))

#tokenisation
def create_tokenizer(lines):
    tokenizer=Tokenizer(lower=True,filters=' ')
    tokenizer.fit_on_texts(lines)
    print(tokenizer.word_index.items())
    return tokenizer

#max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

#encoding and padding sequences
def encoding_sequences(tokenizer, length,lines):
    x=tokenizer.texts_to_sequences(lines)
    x=pad_sequences(x,maxlen=length,padding='post')
    return x

#mapping integer to a word
def word_for_index(integer,tokenizer):
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
        word=word_for_index(i,tokenizer)
        if word is None:
            break
        obj.append(word)
    return ' '.join(obj)


#evaluation
def evaluate_model(model,tokenizer,subjects,raw_dataset):
    actual,predicted=list(),list()
    for i,source in enumerate(subjects):
        #translating
        source=source.reshape((1,source.shape[0]))
        translation=predict_obj(model,tokenizer,source)
        raw_sub,raw_obj=raw_dataset[i]
        if i<10:
            print('source=%s,target=%s,predicted=%s' %(raw_sub,raw_obj,translation))
        actual.append(raw_obj.split())
        predicted.append(translation.split())
    #bleu scores
    print("bleu_scores:")
    print('BLUE_1: %f' % corpus_bleu(actual, predicted))

'''    
    print('BLUE_1: %f' % corpus_bleu(actual, predicted, weights=(1.0,0,0,0)))
    print('BLUE_2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLUE_3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLUE_4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
    '''

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

#preparing data
trainx=encoding_sequences(sub_tokenizer,sub_length,train[:,0])
print(trainx.shape)
testx=encoding_sequences(sub_tokenizer,sub_length,test[:,0])
print(trainx.shape)


#loading the model:
model=load_model('model_13.h5')

print('training')
evaluate_model(model,obj_tokenizer,trainx,train)

print("testing")
evaluate_model(model,obj_tokenizer,testx,test)
