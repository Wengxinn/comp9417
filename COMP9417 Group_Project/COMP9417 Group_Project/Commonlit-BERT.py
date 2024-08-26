#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 11:44:49 2021

@author: MaggieGuan
"""

# Inspired by and reference to: https://colab.research.google.com/drive/1cV516YJdolaABHgkBUoI0mMhN08tZkga?usp=sharing


import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from transformers import TFBertModel, BertConfig, BertTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
if not os.path.exists("/content/drive/MyDrive/checkpoint"):
  os.mkdir('/content/drive/MyDrive/checkpoint')
  
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(23)

#read in data and split into train/test set
train_path = '/content/drive/MyDrive/train.csv'
train_df = pd.read_csv(train_path)
train_df, test_df = train_test_split(train_df,test_size=0.3,random_state=42)

#import bert pre-trained model
bert_model = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(bert_model)

# config will be downloaded and cached
model_config = BertConfig.from_pretrained(bert_model)
model_config.output_hidden_states = True
# Downloads the model 
bert = TFBertModel.from_pretrained(bert_model, config = model_config)

# callbacks
ckpt_dir = '/content/drive/MyDrive/checkpoint/ckpt{epoch:02d}.h5'
ckpt = ModelCheckpoint(
    filepath = ckpt_dir,
    save_freq = 'epoch',
    save_weights_only=True)

loss = 'mse'
metrics = [RootMeanSquaredError()]
callbacks = [ckpt]

def encoder(df, tokenizer, label = 'excerpt', maxLen = 210):
    input_id = []
    token_type = []
    attention_mask = []
    for i in df[label].values:
        token = tokenizer(i, max_length = maxLen, truncation = True, padding = 'max_length', add_special_tokens = True)
        input_id.append(token['input_ids'])
        token_type.append(token['token_type_ids'])
        attention_mask.append(token['attention_mask'])
    return np.array(input_id), np.array(token_type), np.array(attention_mask)
##tunable parameters
## max_len: 180, 200, 220
## epochs: 2, 3, 4
## learning_rate: 
learning_rate = 2e-5
epochs = 3
max_len = 220

#tokenise train and test dataset
train_d = encoder(df=train_df, tokenizer=tokenizer, maxLen=max_len)
test_d = encoder(df=test_df, tokenizer=tokenizer, maxLen=max_len)

optimizer = Adam(learning_rate = learning_rate)


input_ids_i = Input(shape = (max_len, ), dtype = tf.int32, name = 'input_ids')
token_type_ids_i = Input(shape = (max_len, ), dtype = tf.int32, name = 'token_type_ids')
attention_mask_i = Input(shape = (max_len, ), dtype = tf.int32, name = 'attention_mask')
inputs = [input_ids_i, token_type_ids_i, attention_mask_i]

bert_output = bert(input_ids_i, token_type_ids = token_type_ids_i, attention_mask = attention_mask_i)[0]
output = bert_output[:, 0, :]

output = Dropout(0.1)(output)

output = Dense(10, activation = 'linear')(output)
output = Dense(1, activation = 'linear')(output)

model = Model(inputs = inputs, outputs = output)

model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

model.summary()

#split into test and validation set
train_l = train_df['target'].values
val_prob = 0.1
split = int(len(train_l)*(1 - val_prob))
train_x = tuple(np.array(train_d)[:, :split, :])
train_y = train_l[:split]
val_x = tuple(np.array(train_d)[:, split:, :])
val_y = np.array(train_l[split:])

#train model on the target values
model.fit(train_x, train_y, validation_data = (val_x, val_y), epochs = epochs, callbacks = callbacks)
model.save_weights('/content/drive/MyDrive/BERTv1.h5')

#get test errors
print('Test error: ', np.sqrt(mean_squared_error(test_df.target, model.predict(test_d))))
test_df['predict'] = model.predict(test_d)
test_df.to_csv("/test_prediction_bert.csv",index=False)