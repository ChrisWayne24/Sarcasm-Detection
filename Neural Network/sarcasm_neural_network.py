import pandas as pd
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold

#count 4000, acc=84.44 (NoEdit dataset)
#count 7000, acc=85 (NoEdit dataset)

#count 4000, acc=83.92 (NLP preprocessed dataset)
#count 7000, acc=84.3 (NLP preprocessed dataset)

#count 4000, acc=84.16 (LowPlusNoPunc dataset) Lower case plus no Punctuation dataset
#count 5000, acc=84.39 (LowPlusNoPunc dataset)
#count 6000, acc=84.67 (LowPlusNoPunc dataset)
#count 7000, acc=84.85 (LowPlusNoPunc dataset)

count=7000

#Load dataset
d = pd.read_csv("Pre-processed_Sarcasm_Headlines_Dataset.csv")
d = d.sample(frac=1)

def train_and_test(train_idx, test_idx, count):
    #Split into train and test
    train_content = d['headline'].iloc[train_idx]
    test_content = d['headline'].iloc[test_idx]
    
    tokenizer = Tokenizer(num_words=count)
    
    # learn the training words (not the testing words!)
    tokenizer.fit_on_texts(train_content)

    # options for mode: binary, freq, tfidf
    d_train_inputs = tokenizer.texts_to_matrix(train_content, mode='tfidf')
    d_test_inputs = tokenizer.texts_to_matrix(test_content, mode='tfidf')

    # divide tfidf by max
    d_train_inputs = d_train_inputs/np.amax(np.absolute(d_train_inputs))
    d_test_inputs = d_test_inputs/np.amax(np.absolute(d_test_inputs))

    # subtract mean, to get values between -1 and 1
    d_train_inputs = d_train_inputs - np.mean(d_train_inputs)
    d_test_inputs = d_test_inputs - np.mean(d_test_inputs)

    # one-hot encoding of outputs
    d_train_outputs = np_utils.to_categorical(d['is_sarcastic'].iloc[train_idx])
    d_test_outputs = np_utils.to_categorical(d['is_sarcastic'].iloc[test_idx])

    #ML neural network model
    model = Sequential()
    model.add(Dense(512, input_shape=(count,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adamax',
                  metrics=['accuracy'])

    model.fit(d_train_inputs, d_train_outputs, epochs=10, batch_size=128)

    scores = model.evaluate(d_test_inputs, d_test_outputs)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return scores

#CROSS Validation
kfold = StratifiedKFold(n_splits=5)
splits = kfold.split(d, d['is_sarcastic'])
cvscores = []
for train_idx, test_idx, in splits:
    scores = train_and_test(train_idx, test_idx, count)
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
