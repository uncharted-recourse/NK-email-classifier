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

def prepare_data(datapaths, labels):
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

ham_datapaths = ["data/enron.jsonl",
                "dry_run_data/historical_chris.jsonl",
                "dry_run_data/historical_christine.jsonl",
                "dry_run_data/historical_ian.jsonl",
                "dry_run_data/historical_paul.jsonl",
                "dry_run_data/historical_wayne.jsonl"]
spam_datapaths = ["data/nigerian.jsonl",
                "data/Malware.jsonl",
                "data/CredPhishing.jsonl",
                "data/PhishTraining.jsonl",
                "data/Propaganda.jsonl",
                "data/SocialEng.jsonl",
                "data/Spam.jsonl",
                "ta3-attacks/ta3-may-campaign.jsonl",
                "ta3-attacks/ta3-june-campaign.jsonl",
                "ta3-attacks/ta3-july-campaign.jsonl"]

# header = []
# raw_data = None
# raw_data, header = prepare_data(ham_datapaths, ['friend'])
# raw_data, header = prepare_data(spam_datapaths, ['foe'])

# # transpose the data, make everything lower case string
# raw_data = np.char.lower(np.transpose(raw_data).astype('U'))

# save data for future experiments
# f = open('header', 'wb')
# np.save(f, header)
# f.close()
# f = open('raw_data', 'wb')
# np.save(f, raw_data)
# f.close()

header = np.load('header', allow_pickle=True)
raw_data = np.load('raw_data', allow_pickle=True)

# set up appropriate data encoder
Categories = ['friend','foe']
header = np.load('header', allow_pickle=True)
raw_data = np.load('raw_data', allow_pickle=True)
encoder = Encoder(categories=Categories)
encoder.process(raw_data, max_cells)
# encode the data 
X, y, class_weights = encoder.encode_data(raw_data, header, maxlen)
print(class_weights)
print(X.shape)
print(y.shape)

# setup classifier, compile model appropriately
Classifier = Simon(encoder=encoder)
data = Classifier.setup_test_sets(X, y)
model = Classifier.generate_model(maxlen, max_cells, 2,activation='softmax')
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['binary_accuracy'])

# train model
checkpoint_dir = "checkpoint_ta3_attack/"
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
start = time.time()
history = Classifier.train_model(model, data, checkpoint_dir, class_weight=class_weights, batch_size=24)
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

test_header = []
test_raw_data = None
test_raw_data, test_header = prepare_data(test_friend_datapaths, ['friend'])
test_raw_data, test_header = prepare_data(test_foe_datapaths, ['foe'])

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

'''
# do p_threshold ROC tuning on the test data to see if you can improve it
start = time.time()
p_thresholds = np.linspace(0.01,0.99,num=20)
TPR_arr,FPR_arr = Classifier.tune_ROC_metrics(max_cells, model, data, encoder,p_thresholds)
print("DEBUG::True positive rate w.r.t p_threshold array:")
print(TPR_arr)
print("DEBUG::False positive rate w.r.t p_threshold array:")
print(FPR_arr)
# plot
plt.figure()
plt.subplot(311)
plt.plot(p_thresholds,TPR_arr)
plt.xlabel('p_threshold')
plt.ylabel('TPR')
plt.subplot(312)
plt.xlabel('p_threshold')
plt.ylabel('FPR')
plt.plot(p_thresholds,FPR_arr)
plt.subplot(313)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(FPR_arr,TPR_arr)
plt.show()
# timing info
end = time.time()
print("Time for hyperparameter (per-class threshold) is %f sec"%(end-start))
'''
