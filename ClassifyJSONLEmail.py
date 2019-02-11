import time
import json
import os.path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import nltk.data
from random import shuffle

from Simon import Simon 
from Simon.Encoder import Encoder
from Simon.DataGenerator import DataGenerator

# filename = "/home/azunre/Downloads/jpl-abuse-jsonl/PhishTraining.jsonl"
# filename = "/home/azunre/Downloads/jpl-abuse-jsonl/Propaganda.jsonl"

# filename = "/home/azunre/Downloads/enron.jsonl"
filename = "/home/azunre/Downloads/nigerian.jsonl"

# set important parameters
maxlen = 100 # length of each sentence
max_cells = 500 # maximum number of sentences per email
p_threshold = 0.5 # decision boundary

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
with open(filename) as data_file:
    data_JSONL_lines = data_file.readlines()

N = 2500 # number of samples to draw
# visualize body extraction for first email
idx = 0
sample_email = json.loads(data_JSONL_lines[idx])["body"]
print("DEBUG::sample email (whole, then tokenized into sentences):")
print(sample_email)
sample_email_sentence = tokenizer.tokenize(sample_email)
sample_email_sentence = [elem[-maxlen:] for elem in sample_email_sentence] # truncate
print(sample_email_sentence)
all_email_df = pd.DataFrame(sample_email_sentence,columns=['Email 0'])
# now, build up pandas dataframe of appropriate format for NK email classifier
for line in data_JSONL_lines:
    idx = idx+1
    sample_email = json.loads(line)["body"]
    sample_email_sentence = tokenizer.tokenize(sample_email)
    sample_email_sentence = [elem[-maxlen:] for elem in sample_email_sentence] #truncate
    all_email_df = pd.concat([all_email_df,pd.DataFrame(sample_email_sentence,columns=['Email '+str(idx)])],axis=1)
    if idx>=N-1:
        break

print("DEBUG::DONE Loading JSONL data!!")
print("DEBUG::the final shape is:")
print(all_email_df.shape)

all_email_df = all_email_df.astype(str)
# print("DEBUG::the resulting dataframe is:")
# print(all_email_df)

raw_data = np.asarray(all_email_df.ix[:max_cells-1,:]) #truncate to max_cells
header = [['spam'],]*all_email_df.shape[1]
tmp = np.char.lower(np.transpose(raw_data).astype('U'))
raw_data = tmp

# load checkpoint (encoder with categories, weights)
checkpoint_dir = "saved_checkpoints/01162019_best/"
config = Simon({}).load_config('text-class.18-0.08.pkl',checkpoint_dir)
encoder = config['encoder']
checkpoint = config['checkpoint']
print("DEBUG::CHECKPOINT:")
print(checkpoint)
Categories = encoder.categories
print("DEBUG::Categories:")
print(Categories)
# encode the data 
X, y = encoder.encode_data(raw_data, header, maxlen)
# setup classifier, compile model appropriately
Classifier = Simon(encoder=encoder)
data = type('data_type',(object,),{'X_test':X,'y_test':y})
model = Classifier.generate_model(maxlen, max_cells, 2,activation='softmax')
Classifier.load_weights(checkpoint,config,model,checkpoint_dir)
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['binary_accuracy'])

# evaluate model
Classifier.evaluate_model(max_cells, model, data, encoder, p_threshold)