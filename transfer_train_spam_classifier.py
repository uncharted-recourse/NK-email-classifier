/bin/bash: x: command not found
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
        raw_data = data
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

#ham_datapaths = ["data/enron.jsonl",
ham_datapaths= ["dry_run_data/historical_chris.jsonl",
                "dry_run_data/historical_christine.jsonl",
                "dry_run_data/historical_ian.jsonl",
                "dry_run_data/historical_paul.jsonl",
                "dry_run_data/historical_wayne.jsonl"]
#spam_datapaths = ["data/nigerian.jsonl",
#                "data/Malware.jsonl",
#                "data/CredPhishing.jsonl",
#                "data/PhishTraining.jsonl",
#                "data/Propaganda.jsonl",
#                "data/SocialEng.jsonl",
#                "data/Spam.jsonl",
spam_datapaths= ["ta3-attacks/ta3-may-campaign.jsonl",
                "ta3-attacks/ta3-june-campaign.jsonl",
                "ta3-attacks/ta3-july-campaign.jsonl"]

header = []
raw_data = None
raw_data, header = prepare_data(raw_data, header, ham_datapaths, ['friend'])
raw_data, header = prepare_data(raw_data, header, spam_datapaths, ['foe'])

# # transpose the data, make everything lower case string
raw_data = np.char.lower(np.transpose(raw_data).astype('U'))

# save data for future experiments
# f = open('header', 'wb')
# np.save(f, header)
# f.close()
# f = open('raw_data', 'wb')
# np.save(f, raw_data)
# f.close()

#header = np.load('header', allow_pickle=True)
#raw_data = np.load('raw_data', allow_pickle=True)

# load checkpoint (encoder with categories, weights)
modelName = 'text-class.10-0.03.pkl'
checkpoint_dir = "checkpoint_ta3_attack/"
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

# build classifier model    
Classifier = Simon(encoder=encoder) # text classifier for unit test    
model = Classifier.generate_transfer_model(maxlen, max_cells, 2, category_count, checkpoint, checkpoint_dir, activation='sigmoid', all_trainable = True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

# print all layers to make sure it is right
print("DEBUG::total number of layers:")
print(len(model.layers))

# encode the data and evaluate model
X, y, class_weights = encoder.encode_data(raw_data, header, maxlen)
data = Classifier.setup_test_sets(X, y)
max_cells = encoder.cur_max_cells
start = time.time()
history = Classifier.train_model(model, data, checkpoint_dir, class_weight=class_weights, epochs=10)
end = time.time()
print("Time for training is %f sec"%(end-start))

config = { 'encoder' :  encoder,
            'checkpoint' : Classifier.get_best_checkpoint(checkpoint_dir) }
Classifier.save_config(config, checkpoint_dir)

# evaluate model
test_friend_datapaths = ["dry_run_data/chris.jsonl",
                "dry_run_data/christine.jsonl",
                "dry_run_data/ian.jsonl",
                "dry_run_data/paul.jsonl",
                "dry_run_data/wayne.jsonl"]
test_foe_datapaths = ["dry_run_data/recourse-attacks.jsonl"]

# test_header = []
# test_raw_data = None
# test_raw_data, test_header = prepare_data(test_raw_data, test_header, test_friend_datapaths, ['friend'])
# test_raw_data, test_header = prepare_data(test_raw_data, test_header, test_foe_datapaths, ['foe'])

# # transpose the data, make everything lower case string
# test_raw_data = np.char.lower(np.transpose(raw_data).astype('U'))

# save data for future experiments
# f = open('test_header', 'wb')
# np.save(f, test_header)
# f.close()
# f = open('test_raw_data', 'wb')
# np.save(f, test_raw_data)
# f.close()

test_header = np.load('test_header', allow_pickle=True)
test_raw_data = np.load('test_raw_data', allow_pickle=True)
X, y, class_weights = encoder.encode_data(test_raw_data, test_header, maxlen)
test_data = type('data_type',(object,),{'X_test':X,'y_test':y})    
Classifier.evaluate_model(max_cells, model, test_data, encoder, p_threshold, checkpoint_dir=checkpoint_dir)
