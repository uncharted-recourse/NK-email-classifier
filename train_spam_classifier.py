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

# Extract enron/419 scam/JPL data from JSONL format
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
print(raw_data.shape)
# raw_data = np.column_stack((raw_data,np.asarray(falsepositives.ix[:max_cells-1,:].sample(n=N_fp,replace=True,axis=1))))
# header.extend([['friend'],]*N_fp)
raw_data = np.column_stack((raw_data,np.asarray(nigerian_prince.ix[:max_cells-1,:].sample(n=N_spam,replace=False,axis=1))))
header.extend([['foe'],]*N_spam)
print(raw_data.shape)
raw_data = np.column_stack((raw_data,np.asarray(malware.ix[:max_cells-1,:].sample(n=N_spam,replace=False,axis=1))))
header.extend([['foe'],]*N_spam)
print(raw_data.shape)
raw_data = np.column_stack((raw_data,np.asarray(credphishing.ix[:max_cells-1,:].sample(n=N_spam,replace=False,axis=1))))
header.extend([['foe'],]*N_spam)
print(raw_data.shape)
print(phishtraining.shape)
raw_data = np.column_stack((raw_data,np.asarray(phishtraining.ix[:max_cells-1,:].sample(n=N_spam,replace=False,axis=1))))
header.extend([['foe'],]*N_spam)
print(raw_data.shape)
raw_data = np.column_stack((raw_data,np.asarray(propaganda.ix[:max_cells-1,:].sample(n=N_spam,replace=True,axis=1))))
header.extend([['foe'],]*N_spam)
print(raw_data.shape)
raw_data = np.column_stack((raw_data,np.asarray(socialeng.ix[:max_cells-1,:].sample(n=N_spam,replace=False,axis=1))))
header.extend([['foe'],]*N_spam)
print(raw_data.shape)
raw_data = np.column_stack((raw_data,np.asarray(spam.ix[:max_cells-1,:].sample(n=N_spam,replace=False,axis=1))))
header.extend([['foe'],]*N_spam)

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

# set up appropriate data encoder
Categories = ['friend','foe']
encoder = Encoder(categories=Categories)
encoder.process(raw_data, max_cells)
# encode the data 
X, y = encoder.encode_data(raw_data, header, maxlen)
# setup classifier, compile model appropriately
Classifier = Simon(encoder=encoder)
data = Classifier.setup_test_sets(X, y)
model = Classifier.generate_model(maxlen, max_cells, 2,activation='softmax')
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['binary_accuracy'])
# train model
batch_size = 64
nb_epoch = 20
checkpoint_dir = "checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
start = time.time()
history = Classifier.train_model(batch_size, checkpoint_dir, model, nb_epoch, data)
end = time.time()
print("Time for training is %f sec"%(end-start))
config = { 'encoder' :  encoder,
            'checkpoint' : Classifier.get_best_checkpoint(checkpoint_dir) }
Classifier.save_config(config, checkpoint_dir)
Classifier.plot_loss(history)
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