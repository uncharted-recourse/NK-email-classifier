import time
import json
import random
import os.path
import nltk.data
import numpy as np
import pandas as pd
import configparser
from matplotlib import pyplot as plt
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
    
    return pd.DataFrame.from_records(DataLengthStandardizerRaw(all_email_df,max_cells))


# set important parameters
maxlen = 200 # max length of each sentence
max_cells = 100 # maximum number of sentences per email
p_threshold = 0.5 # decision boundary
batch_size = 64
nb_epoch = 20

# Extract enron/419 scam/JPL abuse data from JSONL format
N = 7000 # number of samples to draw
datapath = "/home/azunre/Downloads/enron.jsonl"
enron_data = LoadJSONLEmails(N=N,datapath=datapath)
# N_fp = 1000 # number of samples to draw
# datapath = "/home/azunre/Downloads/jpl-abuse-jsonl/FalsePositive.jsonl"
# falsepositives = LoadJSONLEmails(N=N_fp,datapath=datapath)
N_spam = 1000 # number of samples to draw
datapath = "/home/azunre/Downloads/nigerian.jsonl"
nigerian_prince = LoadJSONLEmails(N=N_spam,datapath=datapath)
datapath = "/home/azunre/Downloads/jpl-abuse-jsonl/Malware.jsonl"
malware = LoadJSONLEmails(N=N_spam,datapath=datapath)
datapath = "/home/azunre/Downloads/jpl-abuse-jsonl/CredPhishing.jsonl"
credphishing = LoadJSONLEmails(N=N_spam,datapath=datapath)
datapath = "/home/azunre/Downloads/jpl-abuse-jsonl/PhishTraining.jsonl"
phishtraining = LoadJSONLEmails(N=N_spam,datapath=datapath)
datapath = "/home/azunre/Downloads/jpl-abuse-jsonl/Propaganda.jsonl"
propaganda = LoadJSONLEmails(N=N_spam,datapath=datapath)
datapath = "/home/azunre/Downloads/jpl-abuse-jsonl/SocialEng.jsonl"
socialeng = LoadJSONLEmails(N=N_spam,datapath=datapath)
datapath = "/home/azunre/Downloads/jpl-abuse-jsonl/Spam.jsonl"
spam = LoadJSONLEmails(N=N_spam,datapath=datapath)
# keep dataset approximately balanced
raw_data = np.asarray(enron_data.sample(n=N,replace=False,axis=1).ix[:max_cells-1,:])
header = [['friend'],]*N
# raw_data = np.column_stack((raw_data,np.asarray(falsepositives.ix[:max_cells-1,:].sample(n=N_fp,replace=True,axis=1))))
# header.extend([['friend'],]*N_fp)
raw_data = np.column_stack((raw_data,np.asarray(nigerian_prince.ix[:max_cells-1,:].sample(n=N_spam,replace=False,axis=1))))
header.extend([['419_scam'],]*N_spam)
raw_data = np.column_stack((raw_data,np.asarray(malware.ix[:max_cells-1,:].sample(n=N_spam,replace=False,axis=1))))
header.extend([['malware'],]*N_spam)
raw_data = np.column_stack((raw_data,np.asarray(credphishing.ix[:max_cells-1,:].sample(n=N_spam,replace=False,axis=1))))
header.extend([['credphishing'],]*N_spam)
raw_data = np.column_stack((raw_data,np.asarray(phishtraining.ix[:max_cells-1,:].sample(n=N_spam,replace=False,axis=1))))
header.extend([['phishtraining'],]*N_spam)
raw_data = np.column_stack((raw_data,np.asarray(propaganda.ix[:max_cells-1,:].sample(n=N_spam,replace=True,axis=1))))
header.extend([['propaganda'],]*N_spam)
raw_data = np.column_stack((raw_data,np.asarray(socialeng.ix[:max_cells-1,:].sample(n=N_spam,replace=False,axis=1))))
header.extend([['socialeng'],]*N_spam)
raw_data = np.column_stack((raw_data,np.asarray(spam.ix[:max_cells-1,:].sample(n=N_spam,replace=False,axis=1))))
header.extend([['spam'],]*N_spam)

print("DEBUG::final labeled data shape:")
print(raw_data.shape)
print(raw_data)


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

# load checkpoint (encoder with categories, weights)
config = configparser.ConfigParser()
config.read('config.ini')
modelName = config['DEFAULT']['modelName']
Categories = ['friend','419_scam','malware','credphishing','phishtraining','propaganda','socialeng','spam']
checkpoint_dir = "deployed_checkpoints/"
config = Simon({}).load_config(modelName,checkpoint_dir)
encoder = config['encoder']
checkpoint = config['checkpoint']
encoder.categories=Categories
category_count = len(Categories)
# print some debug info
print("DEBUG::Categories:")
print(encoder.categories)
    
# build classifier model    
Classifier = Simon(encoder=encoder) # text classifier for unit test    
model = Classifier.generate_transfer_model(maxlen, max_cells, 2, category_count, checkpoint, checkpoint_dir, activation='softmax')

model_compile = lambda m: m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['binary_accuracy'])
model_compile(model)

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