# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 00:25:35 2023

@author: shubh
"""
from tensorflow.keras.models import load_model
import numpy as np
import os
import pickle
def change_working_directory():
    path = input("Please enter the directory and ensure that chrome driver is in the directory (without quotes): ")
    os.chdir(path)
    wd = os.getcwd()
    print(f"Changed working directory to: {os.getcwd()}")
    return wd
wd = change_working_directory()

with open('tokenizer_1000.pkl', 'rb') as f:
    tokenizer_1000 = pickle.load(f)
    
model = load_model('model_1000.h5',compile = False)

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
predicted_word = Predict_Next_Word(model, tokenizer_1000, sentence)
print(f"The predicted next word is: {predicted_word}")
