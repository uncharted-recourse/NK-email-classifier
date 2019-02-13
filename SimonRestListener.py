import nltk.data
from random import shuffle
from json import JSONEncoder
from flask import Flask, request

import time
import pandas
import pickle
import numpy as np
import configparser

import os.path
import pandas as pd

from Simon import *
from Simon.Encoder import *
from Simon.DataGenerator import *
from Simon.LengthStandardizer import *




class SimonRestListener:
    """ SimonRestListener accepts an email, and
    uses its predictive model to generate a score for it - is it
    "friend" or "foe"?
    """
    def __init__(self, modelName):
       self.model =  modelName
       self.encoder = JSONEncoder()

    def runModel(self, sample_email, p_threshold):

        start_time = time.time()
        # set important parameters
        maxlen = 200 # length of each sentence
        max_cells = 100 # maximum number of sentences per email

        DEBUG = True # boolean to specify whether or not print DEBUG information

        checkpoint_dir = "/clusterfiles/saved_checkpoints/02122019/"

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        
        print("DEBUG::sample email (whole, then tokenized into sentences):")
        print(sample_email)
        sample_email_sentence = tokenizer.tokenize(sample_email)
        sample_email_sentence = [elem[-maxlen:] for elem in sample_email_sentence] # truncate
        print(sample_email_sentence)
        all_email_df = pd.DataFrame(sample_email_sentence,columns=['Email 0'])
        print("DEBUG::the final shape is:")
        print(all_email_df.shape)
        all_email_df = all_email_df.astype(str)
        raw_data = np.asarray(all_email_df.ix[:max_cells-1,:]) #truncate to max_cells
        raw_data = np.char.lower(np.transpose(raw_data).astype('U'))

        execution_config=modelName

        # load specified execution configuration
        if execution_config is None:
            raise TypeError
        Classifier = Simon(encoder={}) # dummy text classifier

        config = Classifier.load_config(execution_config, checkpoint_dir)
        encoder = config['encoder']
        checkpoint = config['checkpoint']

        X = encoder.x_encode(raw_data,maxlen)


        # orient the user a bit
        print("DEBUG::CHECKPOINT:")
        print(checkpoint)
        Categories = encoder.categories
        print("DEBUG::Categories:")
        print(Categories)
        category_count = len(Categories)

        model = Classifier.generate_model(maxlen, max_cells, 2,activation='softmax')
        Classifier.load_weights(checkpoint,config,model,checkpoint_dir)
        model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['binary_accuracy'])

        y = model.predict(X)
        # discard empty column edge case
        y[np.all(all_email_df.isnull(),axis=0)]=0

        result = encoder.reverse_label_encode(y,p_threshold)

        elapsed_time = time.time()-start_time
        print("Total time for classification is : %.2f sec" % elapsed_time)
        
        Classifier.clear_session()  # critical for enabling repeated calls of function
        
        return self.encoder.encode((result))

    def predict(self, request_data,p_threshold):
        email = pickle.loads(request_data)
                
        return self.runModel(email,p_threshold)
        
config = configparser.ConfigParser()
config.read('config.ini')
modelName = config['DEFAULT']['modelName']

# modelName = "text-class.17-0.07.pkl" # consider specifying explicitly before building each docker image
print("using model " + modelName + " ...")
        
listener = SimonRestListener(modelName)

app = Flask(__name__)

@app.route("/", methods=['POST'])
def predict():
    """ Listen for data being POSTed on root. The data expected to 
    be a string representation of an email to be classified/scored/analyzed
    """
    request.get_data()
    p_threshold=0.5 # this is potentially tunable
    return listener.predict(request.data,p_threshold)

# $env:FLASK_APP="rest/SimonRestListener.py"
# flask run