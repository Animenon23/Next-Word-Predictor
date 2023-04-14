# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 00:14:44 2023

@author: shubh
"""


import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
import os

def change_working_directory():
    path = input("Please enter the directory and ensure that chrome driver is in the directory (without quotes): ")
    os.chdir(path)
    wd = os.getcwd()
    print(f"Changed working directory to: {os.getcwd()}")
    return wd
wd = change_working_directory()
df = pd.read_csv('IMDB Dataset.csv',nrows = 1000)

def preprocess_text(text):
    # Replace <br> tags with a space
    text = re.sub('<br\s*/?>', ' ', text)
    # Remove HTML tags and punctuation marks
    text = re.sub('<[^>]*>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(' +', ' ',text)
    text = re.sub("\d+", "", text)
    # Convert to lowercase
    text = text.lower()
    return text

df['review'] = df['review'].apply(preprocess_text)
text = ' '.join(df['review'])
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequence_data = tokenizer.texts_to_sequences([text])[0]
vocab_size = len(tokenizer.word_index) + 1
sequences = []

for i in range(1, len(sequence_data)):
    words = sequence_data[i-1:i+1]
    sequences.append(words)
X = []
y = []

for i in sequences:
    X.append(i[0])
    y.append(i[1])
    
X = np.array(X)
y = np.array(y)
y = to_categorical(y, num_classes=vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))

checkpoint = ModelCheckpoint("nextword1.h5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto')

reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)

logdir='logsnextword1'
tensorboard_Visualization = TensorBoard(log_dir=logdir)

model.compile(loss="categorical_crossentropy", optimizer='adam',metrics = ['accuracy'])

model.fit(X, y, epochs=150, batch_size=64, callbacks=[checkpoint, reduce, tensorboard_Visualization])

def Predict_Next_Word(model, tokenizer, text):
    text = text.lower().strip()
    text = text.split(' ')
    text = text[-1]
    text = ''.join(text)
    sequence = tokenizer.texts_to_sequences([text])
    
    if not sequence:
        return 'No prediction'
    
    sequence = np.array(sequence)
    
    if sequence.size == 0:
        return 'No prediction'
    
    preds = model.predict(sequence)[0]
    top_pred_idx = np.argmax(preds)
    predicted_word = tokenizer.index_word[top_pred_idx]
    return predicted_word

sentence = input("Enter a word/sentence: ")
predicted_word = Predict_Next_Word(model, tokenizer, sentence)
print(f"The predicted next word is: {predicted_word}")

#Prediction on train set:
df_test = pd.read_csv('IMDB Dataset.csv',nrows=1000)

# Preprocess the text
df_test['review'] = df_test['review'].apply(preprocess_text)
text_test = ' '.join(df_test['review'])

# Tokenize the text and generate sequences
sequence_data_test = tokenizer.texts_to_sequences([text_test])[0]

# Split the sequences into input and output
sequences_test = []
for i in range(1, len(sequence_data_test)):
    words = sequence_data_test[i-1:i+1]
    sequences_test.append(words)

X_test = []
y_test = []

for i in sequences_test:
    X_test.append(i[0])
    y_test.append(i[1])

X_test = np.array(X_test)
y_test = np.array(y_test)
y_test = to_categorical(y_test, num_classes=vocab_size)

# Generate predictions for each review and calculate accuracy
correct = 0
for i in range(len(X_test)):
    pred = model.predict(X_test[i].reshape(1, -1))[0]
    top_pred_idx = np.argmax(pred)
    predicted_word = tokenizer.index_word[top_pred_idx]
    actual_word = tokenizer.index_word[y_test[i].argmax()]
    print(f'Actual word: {actual_word}')
    print(f'Predicted word: {predicted_word}')
    if predicted_word == actual_word:
        correct += 1

accuracy = correct / len(X_test)
print(f"Accuracy: {accuracy}")

#Prediction on test set:
# Load the next 100 reviews
df_test = pd.read_csv('IMDB Dataset.csv', skiprows=range(1,1001),nrows=1000)

# Preprocess the text
df_test['review'] = df_test['review'].apply(preprocess_text)
text_test = ' '.join(df_test['review'])

# Tokenize the text and generate sequences
sequence_data_test = tokenizer.texts_to_sequences([text_test])[0]

# Split the sequences into input and output
sequences_test = []
for i in range(1, len(sequence_data_test)):
    words = sequence_data_test[i-1:i+1]
    sequences_test.append(words)

X_test = []
y_test = []

for i in sequences_test:
    X_test.append(i[0])
    y_test.append(i[1])

X_test = np.array(X_test)
y_test = np.array(y_test)
y_test = to_categorical(y_test, num_classes=vocab_size)

# Generate predictions for each review and calculate accuracy
correct = 0
for i in range(len(X_test)):
    pred = model.predict(X_test[i].reshape(1, -1))[0]
    top_pred_idx = np.argmax(pred)
    predicted_word = tokenizer.index_word[top_pred_idx]
    actual_word = tokenizer.index_word[y_test[i].argmax()]
    print(f'Actual word: {actual_word}')
    print(f'Predicted word: {predicted_word}')
    if predicted_word == actual_word:
        correct += 1

accuracy = correct / len(X_test)
print(f"Accuracy: {accuracy}")