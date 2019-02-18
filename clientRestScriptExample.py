import os.path
import numpy as np

import pickle
import requests
import json
from json import JSONDecoder


# Load sample email to be classified
filename = "/home/azunre/Downloads/jpl-abuse-jsonl/PhishTraining.jsonl"
# filename = "/home/azunre/Downloads/jpl-abuse-jsonl/Propaganda.jsonl"
# filename = "/home/azunre/Downloads/enron.jsonl"
# filename = "/home/azunre/Downloads/nigerian.jsonl"
with open(filename) as data_file:
    data_JSONL_lines = data_file.readlines()
idx = 47
sample_email = json.loads(data_JSONL_lines[idx])["body"]
print("DEBUG::sample email:")
print(sample_email)        
decoder = JSONDecoder()
address = 'http://localhost:5000/'
r = requests.post(address, data = pickle.dumps(sample_email))
result = decoder.decode(r.text)
print("DEBUG::result:")
print(result)