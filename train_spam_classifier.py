import time
import os.path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Simon import Simon 
from Simon.Encoder import Encoder
from Simon.DataGenerator import DataGenerator

# set important parameters
maxlen = 100 # length of each sentence
max_cells = 500 # maximum number of sentences per email
p_threshold = 0.5 # decision boundary

# define a pandas converter to truncate long cells
def auto_truncate(val):
    return val[-maxlen:]

# read in different data files
datapath = '/vectorizationdata/enron_emails/preprocessed_enron_emails_batch_1.csv'
tmp = pd.read_csv(datapath,dtype=str,encoding="ISO-8859-1",header=0)
enron_batch_1 = pd.read_csv(datapath,dtype=str,encoding="ISO-8859-1",converters={k:auto_truncate for k in range(tmp.shape[1])},header=0)
datapath = '/vectorizationdata/enron_emails/preprocessed_enron_emails_batch_2.csv'
tmp = pd.read_csv(datapath,dtype=str,encoding="ISO-8859-1",header=0)
enron_batch_2 = pd.read_csv(datapath,dtype=str,encoding="ISO-8859-1",converters={k:auto_truncate for k in range(tmp.shape[1])},header=0)
datapath = '/vectorizationdata/nigerian_prince/preprocessed_nigerian_prince_emails.csv'
tmp = pd.read_csv(datapath,dtype=str,encoding="ISO-8859-1",header=0)
nigerian_prince = pd.read_csv(datapath,dtype=str,encoding="ISO-8859-1",converters={k:auto_truncate for k in range(tmp.shape[1])},header=0)


N = 800 # number of ham samples to draw from each of 2 enron batches (spam is 2*N for balance)

raw_data = np.asarray(enron_batch_1.sample(n=N,replace=False,axis=1).ix[:max_cells-1,:])
header = [['ham'],]*N
raw_data = np.column_stack((raw_data,np.asarray(enron_batch_2.sample(n=N,replace=False,axis=1).ix[:max_cells-1,:])))
header.extend([['ham'],]*N)

n_ham = raw_data.shape[1]

print("DEBUG::n_ham")
print(n_ham)

# keep dataset balanced
raw_data = np.column_stack((raw_data,np.asarray(nigerian_prince.ix[:max_cells-1,:].sample(n=n_ham,replace=False,axis=1))))
header.extend([['spam'],]*n_ham)

print("DEBUG::final labeled data shape:")
print(raw_data.shape)
print(raw_data)


# grow list column by column to catch troublesome ones?
# processed_raw_data = np.asarray(raw_data[:,0]).astype('U')[np.newaxis].T
# for i in range(1,raw_data.shape[1]+1):
#     print(i)
#     print(processed_raw_data.shape)
#     try:
#         processed_raw_data = np.append(processed_raw_data,np.asarray(raw_data[:,i]).astype('U')[np.newaxis].T,1)
#     except:
#         print("failed column "+"i")
# 
# or preallocate?
# processed_raw_data_2 = np.zeros(raw_data.shape,dtype=str)
# for i in range(0,raw_data.shape[1]+1):
#     print(i)
#     print(processed_raw_data_2.shape)
#     try:
#         processed_raw_data_2[:,[i]] = np.asarray(raw_data[:,i]).astype('U')[np.newaxis].T
#     except:
#         print("failed column "+"i")


# transpose the data, make everything lower case string
mini_batch = 100 # because of some memory issues, the next step needs to be done in stages
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
Categories = ['ham','spam']
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
batch_size = 5
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