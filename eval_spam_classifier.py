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

# read in different data files
datapath = '/vectorizationdata/enron_emails/preprocessed_enron_emails_batch_1.csv'
enron_batch_1 = pd.read_csv(datapath,dtype=str,encoding="ISO-8859-1",header=0)
datapath = '/vectorizationdata/enron_emails/preprocessed_enron_emails_batch_2.csv'
enron_batch_2 = pd.read_csv(datapath,dtype=str,encoding="ISO-8859-1",header=0)
datapath = '/vectorizationdata/nigerian_prince/preprocessed_nigerian_prince_emails.csv'
nigerian_prince = pd.read_csv(datapath,dtype=str,encoding="ISO-8859-1",header=0)

N = 50 # number of ham samples to draw from each of 2 enron batches (spam is 2*N for balance)

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
checkpoint_dir = "checkpoints/"
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