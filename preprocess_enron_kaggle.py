import time
import numpy as np
import pandas as pd

# from Simon import Simon
# from Simon.Encoder import Encoder

import nltk.data

from random import shuffle


datapath = '/vectorizationdata/enron_emails/kaggle-enron-emails.csv'
all_emails = pd.read_csv(datapath,dtype=str,header=0)

# do some sample processing on a single sentence to make sure it works correctly
idx = 100
print("DEBUG::The shape of the resulting dataframe is:")
print(all_emails.shape)
print("DEBUG::A sample email is:")
print(all_emails.iloc[idx,1].splitlines())
print("DEBUG::The total length of sample email (in characters) is:")
print(len(all_emails.iloc[idx,1]))
print("DEBUG::The sample email (text only, list) is:")
start = all_emails.iloc[idx,1].find("X-FileName")
print(all_emails.iloc[idx,1][start:].splitlines()[1:])
print("DEBUG::The sample email (text only, joined) is:")
sample_email = "".join(all_emails.iloc[idx,1][start:].splitlines()[1:])
print(sample_email)
print("DEBUG::The total length of sample email (text only, in characters) is:")
print(len(sample_email))

# demonstrate that sentence tokenization works
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# print('\n-------\n'.join(tokenizer.tokenize(sample_email)))
sample_email_sentence = tokenizer.tokenize(sample_email)
print(sample_email_sentence)
print(pd.DataFrame(sample_email_sentence,columns=['Sample Email']))


# process all emails (BATCHED)
BATCH = 2 # 1 through 20
batch_size = int(0.05*all_emails.shape[0])
print("DEBUG::will process "+str(batch_size)+" emails")
# start with the first email
start_idx = (BATCH-1)*batch_size
start = all_emails.iloc[start_idx,1].find("X-FileName")
sample_email = "".join(all_emails.iloc[(BATCH-1)*batch_size,1][start:].splitlines()[1:])
sample_email_sentence = tokenizer.tokenize(sample_email)
all_email_df = pd.DataFrame(sample_email_sentence,columns=['Email '+str((BATCH-1)*batch_size)])
# the rest of the emails
for i in range((BATCH-1)*batch_size+1,BATCH*batch_size):
    print("PR0CESSING EMAIL NUMBER "+str(i))
    start = all_emails.iloc[i,1].find("X-FileName")
    sample_email = "".join(all_emails.iloc[i,1][start:].splitlines()[1:])
    sample_email_sentence = tokenizer.tokenize(sample_email)
    # print(sample_email)
    # print(sample_email_sentence)
    all_email_df = pd.concat([all_email_df,pd.DataFrame(sample_email_sentence,columns=['Email '+str(i)])],axis=1)
    # print(all_email_df)

# write preprocessed emails to file
all_email_df.to_csv("/vectorizationdata/enron_emails/preprocessed_enron_emails_batch_"+str(BATCH)+".csv",index=False)