# -*- coding: utf-8 -*-
"""
Created on Tue Apr 01 14:52:40 2018

@author: kulpatil
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano as tf
import keras, json
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
np.random.seed(3)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

#  http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)


print("You have version", tf.__version__)
data = pd.read_csv("Demand_Space_DL.csv")
data.head()
data['tags'].value_counts()
# Split data into train and test
train_posts, test_posts, train_tags, test_tags = train_test_split(data['post'], data['tags'], test_size = 0.20, random_state = 0)
max_words = 2000
tokenize = text.Tokenizer(num_words=max_words, char_level=False, lower = True)
#help text
tokenize.fit_on_texts(train_posts) # only fit on train
'''

# Tokenizers come with a convenient list of words and IDs
dictionary = tokenize.word_index
# Let's save this out so we can use it later
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)
                     
def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in text.texts_to_matrix(text)]                     



allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for info in train_posts:
    wordIndices = convert_text_to_index_array(info)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

'''
                     
                     
x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)
# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

# Converts the labels numbers
num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# Inspect the dimenstions 
print('x_train shape:', train_posts.shape)
print('x_test shape:', test_posts.shape)
print('y_train shape:', train_tags.shape)
print('y_test shape:', test_tags.shape)

batch_size = 16
epochs = 4

# Build the model
DemandSpaceModel = Sequential()
DemandSpaceModel.add(Dense(256, input_shape=(max_words,)))
DemandSpaceModel.add(Activation('relu'))
DemandSpaceModel.add(Dropout(0.5))

DemandSpaceModel.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
#DemandSpaceModel.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))

DemandSpaceModel.add(Dense(num_classes))
DemandSpaceModel.add(Activation('softmax'))

DemandSpaceModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# model.fit trains the model
history = DemandSpaceModel.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs, verbose=1, validation_split=0.2)

# Evaluate the accuracy of our trained model
score = DemandSpaceModel.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# Prediction on individual examples
text_labels = encoder.classes_ 

for i in range(10):
    prediction = DemandSpaceModel.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    print(test_posts.iloc[i][:50], "...")
    print('Actual label:' + test_tags.iloc[i])
    print("Predicted label: " + predicted_label + "\n")

y_softmax = DemandSpaceModel.predict(x_test)

y_test_1d = []
y_pred_1d = []

for i in range(len(y_test)):
    probs = y_test[i]
    index_arr = np.nonzero(probs)
    one_hot_index = index_arr[0].item(0)
    y_test_1d.append(one_hot_index)

for i in range(0, len(y_softmax)):
    probs = y_softmax[i]
    predicted_index = np.argmax(probs)
    y_pred_1d.append(predicted_index)

####################Confusion Matrix################################
cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
plt.figure(figsize=(12,10))
plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
plt.show()


'''
#Saving The Model for Future use
model_json = DemandSpaceModel.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
DemandSpaceModel.save_weights('model.h5')
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# and create a model from that
from keras.models import model_from_json
model = model_from_json(loaded_model_json)
# and weight your nodes with your saved values
model.load_weights('model.h5')
'''

# interactive part
#evalSentence=['Sunsilk Nourishing Soft & Smooth Shampoo 180 ml']
# format your input for the neural net

#input1 = tokenize.texts_to_matrix(np.array(evalSentence))
#pred = DemandSpaceModel.predict(np.array(input1))

##
#pred = DemandSpaceModel.predict(np.array([x_test[55]]))
##
#text_labels
#print("Predicted Label: %s with %f%% confidence" % (text_labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))


####################NOTE THIS FOR PREDICTION################################

pdata = pd.read_csv("Predict_Demand_Space.csv")
pdata.head()
pdata_posts=pdata.T.squeeze()
x_predict = tokenize.texts_to_matrix(pdata_posts)
complete_data=[]
for i in range(len(x_predict)):
    prediction = DemandSpaceModel.predict(np.array([x_predict[i]]))
    score= (np.max(prediction))
    #predicted_label=''
    #if np.max(prediction)> .75:
    predicted_label = text_labels[np.argmax(prediction)]
    #print(pdata_posts.iloc[i][:150], "...")
    #print('Actual label:' + test_tags.iloc[i])
    #print("Predicted label: " + predicted_label + "\n")
    complete_data.append([pdata_posts.iloc[i],predicted_label,score])
import csv    
with open("Prediction Output DL.csv", "w",newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Posts', 'Predicted Tags','Score'])
    writer.writerows(complete_data)

########################
'''import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = max_words
train_posts, test_posts, train_tags, test_tags
(X_train, y_train), (X_test, y_test) = (train_posts,train_tags),(test_posts,test_tags)
# truncate and pad input sequences
max_review_length = 150
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_words))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
'''
'''xa=['Amway Satinique Anti Dandruff Shampoo, 250ml']
ya=['Anti-Hairfall Shampoo 675m']
ca=[['BBLUNT Back To Life Dry Shampoo for Instant Freshness, 125ml','Anti-Hairfall Shampoo 675m']]

a=tokenize.texts_to_matrix(xa)
b=tokenize.texts_to_matrix(ya)
c=tokenize.texts_to_matrix(ca)

c[0]
NP = DemandSpaceModel.predict(c)
#y_softmax = DemandSpaceModel.predict(x_test)
#x_test = tokenize.texts_to_matrix(test_posts)
y_pred_1d=[]


for i in range(0, len(NP)):
    probs = NP[i]
    predicted_index = np.argmax(probs)
    y_pred_1d.append(predicted_index)'''
