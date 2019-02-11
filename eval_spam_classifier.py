import time
import os.path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Simon import Simon 
from Simon.Encoder import Encoder
from Simon.DataGenerator import DataGenerator

# extract the first N samples from jsonl
def LoadJSONLEmails(N=2500,datapath):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    with open(filename) as data_file:
        data_JSONL_lines = data_file.readlines()
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
    
    return all_email_df

# set important parameters
maxlen = 500 # max length of each sentence
max_cells = 100 # maximum number of sentences per email
p_threshold = 0.5 # decision boundary

# Extract enron/nigerian prince data from JSONL format
N = 1000 # number of samples to draw
datapath = "/home/azunre/Downloads/enron.jsonl"
enron_data = LoadJSONLEmails(N=N,datapath=datapath)
N_spam = 1000 # number of samples to draw
datapath = "/home/azunre/Downloads/nigerian.jsonl"
nigerian_prince = LoadJSONLEmails(N=N_spam,datapath=datapath)
# keep dataset balanced
raw_data = np.asarray(enron_data.sample(n=N,replace=False,axis=1).ix[:max_cells-1,:])
header = [['ham'],]*N
raw_data = np.column_stack((raw_data,np.asarray(nigerian_prince.ix[:max_cells-1,:].sample(n=N_spam,replace=False,axis=1))))
header.extend([['spam'],]*N_spam)

print("DEBUG::final labeled data shape:")
print(raw_data.shape)
print(raw_data)


# transpose the data, make everything lower case string
mini_batch = 10 # because of some memory issues, the next step needs to be done in stages
start = time.time()
tmp = np.char.lower(np.transpose(raw_data[:,:mini_batch]).astype('U'))
tmp_header = header[:mini_batch]
for i in range(1,int(raw_data.shape[1]/mini_batch)):
    # print("DEBUG::current shape of loaded text (data,header)")
    # print(tmp.shape)
    # print(len(tmp_header))
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
checkpoint_dir = "checkpoints/Second_90_plus/"
config = Simon({}).load_config('text-class.16-0.25.pkl',checkpoint_dir)
encoder = config['encoder']
checkpoint = config['checkpoint']

print("DEBUG::CHECKPOINT:")
print(checkpoint)

Categories = encoder.categories
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
plt.title('TPR/FPR evolution')
plt.subplot(312)
plt.ylabel('FPR')
plt.plot(p_thresholds,FPR_arr)
plt.subplot(313)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.plot(FPR_arr,TPR_arr)
plt.show()
# timing info
end = time.time()
print("Time for hyperparameter (per-class threshold) is %f sec"%(end-start))