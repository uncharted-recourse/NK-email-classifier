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

# extract the first N samples from jsonl
def LoadJSONLEmails(N=10000000,datapath=None):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    with open(datapath) as data_file:
        data_JSONL_lines = data_file.readlines()
    random.shuffle(data_JSONL_lines)
    # visualize body extraction for first email
    idx = 0
    sample_email = json.loads(data_JSONL_lines[idx])["body"]
    print("DEBUG::the current email type being loaded:")
    print(datapath)
    #print("DEBUG::sample email (whole, then tokenized into sentences):")
    #print(sample_email)
    sample_email_sentence = tokenizer.tokenize(sample_email)
    sample_email_sentence = [elem[-maxlen:] for elem in sample_email_sentence] # truncate
    #print(sample_email_sentence)
    all_email_df = pd.DataFrame(sample_email_sentence,columns=['Email 0'])
    # now, build up pandas dataframe of appropriate format for NK email classifier
    for line in data_JSONL_lines:
        print(idx)
        idx = idx+1
        sample_email = ''
        content = json.loads(line)
        #for url in content["urls"]:
        #    sample_email += url + ' '
        sample_email += content["body"]
        sample_email_sentence = tokenizer.tokenize(sample_email)
        sample_email_sentence = [elem[-maxlen:] for elem in sample_email_sentence] #truncate
        all_email_df = pd.concat([all_email_df,pd.DataFrame(sample_email_sentence,columns=['Email '+str(idx)])],axis=1)
        if idx>=N-1:
            break

    return pd.DataFrame.from_records(DataLengthStandardizerRaw(all_email_df,max_cells))


# set important parameters
maxlen = 200 # max length of each sentence
max_cells = 100 # maximum number of sentences per email
p_threshold = 0.5 # decision boundary
batch_size = 32
nb_epoch = 10
'''
# Extract dry run historical data from jsonl format
datapath = "dry_run_data/historical_chris.jsonl"
chris_data = LoadJSONLEmails(datapath=datapath)
datapath = "dry_run_data/historical_christine.jsonl"
christine_data = LoadJSONLEmails(datapath=datapath)
datapath = "dry_run_data/historical_ian.jsonl"
ian_data = LoadJSONLEmails(datapath=datapath)
datapath = "dry_run_data/historical_paul.jsonl"
paul_data = LoadJSONLEmails(datapath=datapath)
datapath = "dry_run_data/historical_wayne.jsonl"
wayne_data = LoadJSONLEmails(datapath=datapath)

N_spam = 150 # number of samples to draw
datapath = "data/nigerian.jsonl"
nigerian_prince = LoadJSONLEmails(N=N_spam,datapath=datapath)
datapath = "data/Malware.jsonl"
malware = LoadJSONLEmails(N=N_spam,datapath=datapath)
datapath = "data/CredPhishing.jsonl"
credphishing = LoadJSONLEmails(N=N_spam,datapath=datapath)
datapath = "data/PhishTraining.jsonl"
phishtraining = LoadJSONLEmails(N=N_spam,datapath=datapath)
datapath = "data/Propaganda.jsonl"
propaganda = LoadJSONLEmails(N=N_spam,datapath=datapath)
datapath = "data/SocialEng.jsonl"
socialeng = LoadJSONLEmails(N=N_spam,datapath=datapath)
datapath = "data/Spam.jsonl"
spam = LoadJSONLEmails(N=N_spam,datapath=datapath)

# keep dataset approximately balanced
raw_data = np.asarray(nigerian_prince.ix[:max_cells-1,:])
header = [['foe'],]*N_spam
print(raw_data.shape)
raw_data = np.column_stack((raw_data,np.asarray(malware.ix[:max_cells-1,:])))
header.extend([['foe'],]*N_spam)
print(raw_data.shape)
raw_data = np.column_stack((raw_data,np.asarray(credphishing.ix[:max_cells-1,:])))
header.extend([['foe'],]*N_spam)
print(raw_data.shape)
print(phishtraining.shape)
raw_data = np.column_stack((raw_data,np.asarray(phishtraining.ix[:max_cells-1,:])))
header.extend([['foe'],]*N_spam)
print(raw_data.shape)
raw_data = np.column_stack((raw_data,np.asarray(propaganda.ix[:max_cells-1,:])))
header.extend([['foe'],]*N_spam)
print(raw_data.shape)
raw_data = np.column_stack((raw_data,np.asarray(socialeng.ix[:max_cells-1,:])))
header.extend([['foe'],]*N_spam)
print(raw_data.shape)
raw_data = np.column_stack((raw_data,np.asarray(spam.ix[:max_cells-1,:])))
header.extend([['foe'],]*N_spam)

# keep dataset approximately balanced
raw_data = np.column_stack((raw_data,np.asarray(chris_data.ix[:max_cells-1,:])))
header.extend([['friend'],]*chris_data.shape[1])
print(raw_data.shape)
raw_data = np.column_stack((raw_data,np.asarray(christine_data.ix[:max_cells-1,:])))
header.extend([['friend'],]*christine_data.shape[1])
print(raw_data.shape)
raw_data = np.column_stack((raw_data,np.asarray(ian_data.ix[:max_cells-1,:])))
header.extend([['friend'],]*ian_data.shape[1])
print(raw_data.shape)
raw_data = np.column_stack((raw_data,np.asarray(paul_data.ix[:max_cells-1,:])))
header.extend([['friend'],]*paul_data.shape[1])
print(raw_data.shape)
raw_data = np.column_stack((raw_data,np.asarray(wayne_data.ix[:max_cells-1,:])))
header.extend([['friend'],]*wayne_data.shape[1])
print(raw_data.shape)

print("DEBUG::final labeled data shape:")
print(raw_data.shape)
print(raw_data)
raw_data = np.char.lower(np.transpose(raw_data).astype('U'))

# transpose the data, make everything lower case string
mini_batch = 1000 # because of some memory issues, the next step needs to be done in stages
start = time.time()
tmp = np.char.lower(np.transpose(raw_data[:,:mini_batch]).astype('U'))
tmp_header = header[:mini_batch]
for i in range(1,int(raw_data.shape[1]/mini_batch)):
    print("DEBUG::current shape of loaded text (data,header)")
    print(tmp.shape)
    print(len(tmp_header))
    try:
        tmp = np.vstack((tmp,np.char.lower(np.transpose(raw_data[:,i*mini_batch:(i+1)*mini_batch]).astype('U'))))
        tmp_header.extend(header[i*mini_batch:(i+1)*mini_batch])
    except:
        print("failed string standardization on batch number "+str(i))

header = tmp_header
end = time.time()
print("Time for casting data as lower case string is %f sec"%(end-start))
raw_data = tmp
# save data for future experiments
f = open('historical_raw_data.npy', 'wb')
np.save(f, raw_data)
f = open('historical_header.npy', 'wb')
np.save(f, header)
'''
# load data 
raw_data = np.load('historical_raw_data.npy', allow_pickle=True)
header = np.load('historical_header.npy', allow_pickle=True)

# load checkpoint (encoder with categories, weights)
modelName = 'text-class.17-0.14.pkl'
checkpoint_dir = "checkpoints/"
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
model = Classifier.generate_transfer_model(maxlen, max_cells, 2, category_count, checkpoint, checkpoint_dir, activation='sigmoid')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

# print all layers to make sure it is right
print("DEBUG::total number of layers:")
print(len(model.layers))
print("DEBUG::They are:")
for layer in model.layers:
    print(layer)
print("DEBUG::model.summary() is:")
print(model.summary())

# encode the data and evaluate model
X, y = encoder.encode_data(raw_data, header, maxlen)

data = Classifier.setup_test_sets(X, y)

max_cells = encoder.cur_max_cells

start = time.time()
history = Classifier.train_model(batch_size, checkpoint_dir, model, nb_epoch, data)
end = time.time()
print("Time for training is %f sec"%(end-start))

config = { 'encoder' :  encoder,
            'checkpoint' : Classifier.get_best_checkpoint(checkpoint_dir) }
Classifier.save_config(config, checkpoint_dir)
Classifier.plot_loss(history) #comment out on docker images...
    
Classifier.evaluate_model(max_cells, model, data, encoder, p_threshold)

# write evaluation metrics to file for comparison
if not os.path.exists('experiment_metrics.txt'):
     file = open('experiment_metrics.txt', 'w')
     file.close()
file.open('experiment_metrics.txt', 'a')
file.write("baseline classifier with urls: %0.3f\n" % (history.history['val_binary_accuracy']))
file.flush()
file.close()
