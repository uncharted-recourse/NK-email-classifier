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
        raw_data = data
    else:
        data = np.column_stack([LoadJSONLEmails(datapath=p) for p in datapaths])
        raw_data = np.column_stack((raw_data, data))
    if type(labels[0]) != list:
        header.extend([labels] * data.shape[1])
    else:
        header.extend(labels)
    return raw_data, header

def parse_ta3_attack_labels(csv_file, datapath, fill = ['gather_general_info']):

    # read in csv files containing subject and label information
    labels = pd.read_csv(csv_file)
    labels = labels[~labels['Motive'].isnull()]
    
    # create data structure linking subject to (multi) labels
    labels = labels.set_index('Scenario')
    labels['labels'] = [[val for val in lst if str(val) != 'nan'] for lst in labels.values]
    labels_dict = labels.to_dict()['labels']
    
    # dictionary translating csv labels to model annotations
    label_to_annotation = {'Install malware': 'install_malware', 
        'Acquire credentials': 'acquire_credentials', 
        'Acquire PII': 'acquire_pii', 
        'Build trust': 'build_trust', 
        'Gain access to social network': 'access_social_network', 
        'Gather general information': 'gather_general_info', 
        'Get money': 'get_money', 
        'Make appointment': 'access_social_network',
    }

    # compare subject line of json to data structure
    annotations = []
    for idx, line in enumerate(open(datapath).readlines()):
        subject = json.loads(line)['subject']
        found = False
        for key, value in labels_dict.items():
            if key in subject or subject in key:
                labels = [label_to_annotation[v] for v in value]
                annotations.append(labels)
                found = True
                break
        if not found:
            # log index, subject if nothing found in data structure
            # fill with default values
            annotations.append(fill)
    return annotations

# set important parameters
maxlen = 200 # max length of each sentence
max_cells = 100 # maximum number of sentences per email
p_threshold = 0.5 # decision boundary
# data annotations
'''
ham_datapaths = ["data/enron.jsonl",
                "dry_run_data/historical_chris.jsonl",
                "dry_run_data/historical_christine.jsonl",
                "dry_run_data/historical_ian.jsonl",
                "dry_run_data/historical_paul.jsonl",
                "dry_run_data/historical_wayne.jsonl"]
malware = ["data/Malware.jsonl"]
acquire_credentials = ["data/CredPhishing.jsonl"] # also gather_general_info
access_social_network = ["data/SocialEng.jsonl"] # also gather_general_info, build_trust
gather_general_info = ["data/PhishTraining.jsonl"]
fear = ["data/Propaganda.jsonl"]
annoy_recipient = ["data/Spam.jsonl"]
may_attacks = ["ta3-attacks/ta3-may-campaign.jsonl"]
june_attacks = ["ta3-attacks/ta3-june-campaign.jsonl"]
july_attacks = ["ta3-attacks/ta3-july-campaign.jsonl"]
may_annotations = parse_ta3_attack_labels("ta3-attacks/May_Campaign.csv", may_attacks[0], fill = ['gather_general_info', 'install_malware'])
june_annotations = parse_ta3_attack_labels("ta3-attacks/June_Campaign.csv", june_attacks[0])
july_annotations = parse_ta3_attack_labels("ta3-attacks/July_Campaign.csv", july_attacks[0])
header = []
raw_data = None
raw_data, header = prepare_data(raw_data, header, ham_datapaths, ['friend'])
raw_data, header = prepare_data(raw_data, header,malware, ['install_malware'])
raw_data, header = prepare_data(raw_data, header,acquire_credentials, ['acquire_credentials', 'gather_general_info'])
raw_data, header = prepare_data(raw_data, header,access_social_network, ['access_social_network', 'gather_general_info', 'build_trust'])
raw_data, header = prepare_data(raw_data, header,gather_general_info, ['gather_general_info'])
raw_data, header = prepare_data(raw_data, header,fear, ['elicit_fear'])
raw_data, header = prepare_data(raw_data, header,annoy_recipient, ['annoy_recipient'])
raw_data, header = prepare_data(raw_data, header,may_attacks, may_annotations)
raw_data, header = prepare_data(raw_data, header,june_attacks, june_annotations)
raw_data, header = prepare_data(raw_data, header,july_attacks, july_annotations)
# transpose the data, make everything lower case string
raw_data = np.char.lower(np.transpose(raw_data).astype('U'))
#save data for future experiments
f = open('header_multi', 'wb')
np.save(f, header)
f.close()
f = open('raw_data_multi', 'wb')
np.save(f, raw_data)
f.close()
'''
header = np.load('header_multi', allow_pickle=True)
raw_data = np.load('raw_data_multi', allow_pickle=True)

# load checkpoint (encoder with categories, weights)
modelName = 'text-class.10-0.03.pkl'
Categories = ['friend', 'install_malware', 'acquire_credentials', 'annoy_recipient', 'acquire_pii',
            'build_trust', 'elicit_fear', 'access_social_network', 'gather_general_info', 'get_money']
checkpoint_dir = "checkpoint_ta3_attack/"
config = Simon({}).load_config(modelName,checkpoint_dir)
encoder = config['encoder']
checkpoint = config['checkpoint']
encoder.categories = Categories
category_count = len(Categories)
# print some debug info
print("DEBUG::Categories:")
print(encoder.categories)
print("DEBUG::CHECKPOINT:")
print(checkpoint)

# build classifier model    
Classifier = Simon(encoder=encoder) # text classifier for unit test    
model = Classifier.generate_transfer_model(maxlen, max_cells, 2, category_count, checkpoint, checkpoint_dir, activation='sigmoid')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

# print all layers to make sure it is right
print("DEBUG::total number of layers:")
print(len(model.layers))
print(header[-10:])
print(len(header))
print(raw_data.shape)
# encode the data and evaluate model
X, y, class_weights = encoder.encode_data(raw_data, header, maxlen)
print(X.shape)
print(y.shape)
print(class_weights)
data = Classifier.setup_test_sets(X, y)
max_cells = encoder.cur_max_cells
start = time.time()
history = Classifier.train_model(model, data, checkpoint_dir, class_weight=class_weights, batch_size=32)
end = time.time()
print("Time for training is %f sec"%(end-start))

config = { 'encoder' :  encoder,
            'checkpoint' : Classifier.get_best_checkpoint(checkpoint_dir) }
Classifier.save_config(config, checkpoint_dir)

# evaluate model
'''
test_friend_datapaths = ["dry_run_data/chris.jsonl",
                "dry_run_data/christine.jsonl",
                "dry_run_data/ian.jsonl",
                "dry_run_data/paul.jsonl",
                "dry_run_data/wayne.jsonl"]
test_foe_datapaths = ["dry_run_data/recourse-attacks.jsonl"]

test_header = []
test_raw_data = None
test_raw_data, test_header = prepare_data(test_friend_datapaths, ['friend'])
test_raw_data, test_header = prepare_data(test_foe_datapaths, ['foe'])

# transpose the data, make everything lower case string
test_raw_data = np.char.lower(np.transpose(raw_data).astype('U'))

# # save data for future experiments
# f = open('test_header', 'wb')
# np.save(f, test_header)
# f.close()
# f = open('test_raw_data', 'wb')
# np.save(f, test_raw_data)
# f.close()

test_header = np.load('test_header', allow_pickle=True)
test_raw_data = np.load('test_raw_data', allow_pickle=True)

test_data = type('data_type',(object,),{'X_test':test_raw_data,'y_test':test_header})    
Classifier.evaluate_model(max_cells, model, test_data, encoder, p_threshold)
'''
