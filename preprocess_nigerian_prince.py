import time
import numpy as np
import pandas as pd
from sys import exit

import nltk.data


datapath = '/vectorizationdata/nigerian_prince/fradulent_emails.txt'
text_file = open(datapath,"r",encoding="ISO-8859-1")
corpus = text_file.read()
print(str(len(corpus))+" characters in corpus")
# lines = corpus.split("\n")
# print(lines[:200]) # first 200 lines for debugging...

# do some sample processing on a single email to make sure it works correctly
start = corpus.find("Status: O")
next_start = 6+corpus[6:].find("From r")
print(start)
print(next_start)
first_email = corpus[start:next_start]
# some debug info on the first email
print("DEBUG::first email:")
print(first_email.splitlines()[1:])
print("DEBUG::The total length of sample email (in characters) is:")
print(len(first_email))
print("DEBUG::The first email (joined) is:")
sample_email = "".join(first_email.splitlines()[1:])
print(sample_email)
# demonstrate that sentence tokenization works
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# print('\n-------\n'.join(tokenizer.tokenize(sample_email)))
sample_email_sentence = tokenizer.tokenize(sample_email)
print(sample_email_sentence)
print(pd.DataFrame(sample_email_sentence,columns=['Sample Email']))


# process all emails

# start with the first email
all_email_df = pd.DataFrame(sample_email_sentence,columns=['Email 0'])
# the rest of the emails
i=1
while start < len(corpus):
# while i < 5: # debug for 1st 5 emails
    # print(start)
    # print(next_start)
    print("PR0CESSING EMAIL NUMBER "+str(i))
    start = next_start+corpus[next_start:].find("Status:")
    tmp = next_start+6+corpus[next_start+6:].find("From r")
    next_start = tmp
    # print(start)
    # print(next_start)
    sample_email = corpus[start:next_start]
    sample_email = "".join(sample_email.splitlines()[1:])
    sample_email_sentence = tokenizer.tokenize(sample_email)
    # print(sample_email)
    # print(sample_email_sentence)
    all_email_df = pd.concat([all_email_df,pd.DataFrame(sample_email_sentence,columns=['Email '+str(i)])],axis=1)
    # print(all_email_df)
    i = i+1

# write preprocessed emails to file
all_email_df.to_csv("/vectorizationdata/nigerian_prince/preprocessed_nigerian_prince_emails.csv",index=False)