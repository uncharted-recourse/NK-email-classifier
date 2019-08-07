import time
import random
import os.path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import nltk.data
from Simon import Simon 
from Simon.Encoder import Encoder
from Simon.DataGenerator import DataGenerator
from Simon.LengthStandardizer import *

def LoadJSONLEmails(N = 50000, datapath=None):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    start = time.time()
    with open(datapath) as data_file:
        data_JSONL_lines = data_file.readlines()[:N]
    print("DEBUG::the current email type being loaded:")
    print(datapath)
    emails = [json.loads(line) for line in data_JSONL_lines]
    print('Content Loading {} took {} seconds'.format(datapath, time.time() - start))
    lines = []
    for email in emails:
        txt = ''
        for url in email['urls']:
            txt += url + '.\n'
        txt += email['subject'] + '.\n'
        txt += email['body']
        sentences = [sentence[:maxlen] for sentence in tokenizer.tokenize(txt)][:max_cells]
        if len(sentences) == 0:
            continue
        none_count = max_cells - len(sentences)
        sentences = np.pad(sentences, (0,none_count), mode='wrap')
        lines.append(sentences)
    print('Parsing Loading {} took {} seconds'.format(datapath, time.time() - start))
    return np.transpose(lines)

def prepare_data(raw_data, header, datapaths, labels):
    if raw_data is None:
        data = np.column_stack([LoadJSONLEmails(datapath=p) for p in datapaths])
    else:
        data = np.column_stack([LoadJSONLEmails(datapath=p) for p in datapaths])
        raw_data = np.column_stack((raw_data, data))
    if len(labels) == 1:
        header.extend([labels] * data.shape[1])
    else:
        header.extend(labels)
    return raw_data, header

# set important parameters
maxlen = 200 # max length of each sentence
max_cells = 100 # maximum number of sentences per email
p_threshold = 0.5 # decision boundary

# load checkpoint (encoder with categories, weights)
modelName = ''
checkpoint_dir = "deployed_checkpoints/"
config = Simon({}).load_config(modelName,checkpoint_dir)
encoder = config['encoder']
checkpoint = config['checkpoint']
Categories = encoder.categories
category_count = len(Categories)
# print some debug info
print("DEBUG::Categories:")
print(encoder.categories)
print("DEBUG::CHECKPOINT:")
print(checkpoint)

# setup classifier, compile model appropriately
Classifier = Simon(encoder=encoder)
model = Classifier.generate_model(maxlen, max_cells, category_count,activation='softmax')
Classifier.load_weights(checkpoint,config,model,checkpoint_dir)
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['binary_accuracy'])

# evaluate model
test_friend_datapaths = ["dry_run_data/chris.jsonl",
                "dry_run_data/christine.jsonl",
                "dry_run_data/ian.jsonl",
                "dry_run_data/paul.jsonl",
                "dry_run_data/wayne.jsonl"]
test_foe_datapaths = ["dry_run_data/recourse-attacks.jsonl"]

test_header = []
test_raw_data = None
test_raw_data, test_header = prepare_data(test_raw_data, test_header, test_friend_datapaths, ['friend'])
test_raw_data, test_header = prepare_data(test_raw_data, test_header, test_foe_datapaths, ['foe'])

# transpose the data, make everything lower case string
test_raw_data = np.char.lower(np.transpose(raw_data).astype('U'))

# save data for future experiments
f = open('test_header', 'wb')
np.save(f, test_header)
f.close()
f = open('test_raw_data', 'wb')
np.save(f, test_raw_data)
f.close()

# test_header = np.load('test_header', allow_pickle=True)
# test_raw_data = np.load('test_raw_data', allow_pickle=True)

test_data = type('data_type',(object,),{'X_test':test_raw_data,'y_test':test_header})
Classifier.evaluate_model(max_cells, model, test_data, encoder, p_threshold, checkpoint_dir = checkpoint_dir)