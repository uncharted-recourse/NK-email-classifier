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
from sklearn.utils.class_weight import compute_class_weight

# extract the first N samples from jsonl
def LoadJSONLEmails(N=10000000,datapath=None):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    with open(datapath) as data_file:
        data_JSONL_lines = data_file.readlines()
    random.shuffle(data_JSONL_lines)
    idx = 0
    sample_email = json.loads(data_JSONL_lines[idx])["body"]
    print("DEBUG::the current email type being loaded:")
    print(datapath)
    #print("DEBUG::sample email (whole, then tokenized into sentences):")
    sample_email_sentence = tokenizer.tokenize(sample_email)
    sample_email_sentence = [elem[-maxlen:] for elem in sample_email_sentence] # truncate
    all_email_df = pd.DataFrame(sample_email_sentence,columns=['Email 0'])
    # now, build up pandas dataframe of appropriate format for NK email classifier
    for line in data_JSONL_lines:
        print(idx)
        idx = idx+1
        sample_email = ''
        content = json.loads(line)
        for url in content["urls"]:
            sample_email += url + ' '
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
                
ham_data = [np.asarray(LoadJSONLEmails(datapath=h).ix[:max_cells-1,:]) for h in ham_datapaths]
spam_data = [np.asarray(LoadJSONLEmails(datapath=h).ix[:max_cells-1,:]) for h in spam_datapaths]
raw_data_ham = np.column_stack(ham_data)
raw_data_spam = np.column_stack(spam_data)
header = [['friend'],] * raw_data_ham.shape[0]
header.extend([['foe'],] * raw_data_spam.shape[0])
raw_data = np.column_stack(raw_data_ham, raw_data_spam)

print("DEBUG::final labeled data shape:")
print(raw_data.shape)
print(raw_data)

# transpose the data, make everything lower case string
mini_batch = 1000 # because of some memory issues, the next step needs to be done in stages
start = time.time()
tmp = np.char.lower(np.transpose(raw_data[:,:mini_batch]).astype('U'))
tmp_header = header[:mini_batch]
for i in range(1,int(raw_data.shape[1]/mini_batch)):
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
f = open('raw_data', 'wb')
np.save(f, raw_data)
f.close()
f = open('header', 'wb')
np.save(f, header)
f.close()

# load data 
#raw_data = np.load('raw_data.npy', allow_pickle=True)
#header = np.load('header.npy', allow_pickle=True)

# set up appropriate data encoder
Categories = ['friend','foe']
encoder = Encoder(categories=Categories)
encoder.process(raw_data, max_cells)
# encode the data 
X, y, class_weights = encoder.encode_data(raw_data, header, maxlen)
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
history = Classifier.train_model(model, data, checkpoint_dir, class_weight=class_weights)
end = time.time()
print("Time for training is %f sec"%(end-start))
config = { 'encoder' :  encoder,
            'checkpoint' : Classifier.get_best_checkpoint(checkpoint_dir) }
Classifier.save_config(config, checkpoint_dir)
Classifier.evaluate_model(max_cells, model, data, encoder, p_threshold, checkpoint_dir = checkpoint_dir)

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
