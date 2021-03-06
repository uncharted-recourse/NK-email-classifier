#
# GRPC Server for NK Email Classifier
# 
# Uses GRPC service config in protos/grapevine.proto
# 

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

import grpc
import logging
import grapevine_pb2
import grapevine_pb2_grpc
from concurrent import futures


restapp = Flask(__name__)

# GLOBALS
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

DEBUG = True # boolean to specify whether or not print DEBUG information

#-----
class NKEmailClassifier(grapevine_pb2_grpc.ClassifierServicer):

    def __init__(self):

        self.maxlen = 200 # length of each sentence
        self.max_cells = 100 # maximum number of sentences per email

        checkpoint_dir = "deployed_checkpoints/"
        execution_config=MODEL_OBJECT

        # load specified execution configuration
        if execution_config is None:
            raise TypeError
        Classifier = Simon(encoder={}) # dummy text classifier

        config = Classifier.load_config(execution_config, checkpoint_dir)

        self.encoder = config['encoder']
        self.checkpoint = config['checkpoint']

        self.model = Classifier.generate_model(self.maxlen, self.max_cells, len(self.encoder.categories),activation='softmax')
        Classifier.load_weights(self.checkpoint,config,self.model,checkpoint_dir)
        self.model._make_predict_function()

    # Main classify function
    def Classify(self, request, context):

        # init classifier result object
        result = grapevine_pb2.Classification(
            domain=DOMAIN_OBJECT,
            prediction='false',
            confidence=0.0,
            model=CLF_NAME,
            version="0.0.7",
            meta=grapevine_pb2.Meta(),
        )

        # get text from input message
        input_doc = ''
        for url in request.urls:
            input_doc+=url + ' '
        input_doc += request.text

        # Exception cases
        if (len(input_doc.strip()) == 0) or (input_doc is None):
            return result

        # SIMON NK_email_classifier prediction code
        sample_email = input_doc
        start_time = time.time()
        
        # set important parameters
        p_threshold = 0.5 # decision boundary

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        
        if(DEBUG):
            print("DEBUG::sample email (whole, then tokenized into sentences):")
            print(sample_email)
        sample_email_sentence = tokenizer.tokenize(sample_email)
        sample_email_sentence = [elem[-self.maxlen:] for elem in sample_email_sentence] # truncate
        print(sample_email_sentence)
        all_email_df = pd.DataFrame(sample_email_sentence,columns=['Email 0'])
        
        all_email_df = all_email_df.astype(str)
        all_email_df = pd.DataFrame.from_records(DataLengthStandardizerRaw(all_email_df,self.max_cells))

        if(DEBUG):
            print("DEBUG::the final shape is:")
            print(all_email_df.shape)

        raw_data = np.asarray(all_email_df.ix[:self.max_cells-1,:]) #truncate to max_cells
        raw_data = np.char.lower(np.transpose(raw_data).astype('U'))

        X = self.encoder.x_encode(raw_data,self.maxlen)

        # orient the user a bit
        if(DEBUG):
            print("DEBUG::CHECKPOINT:")
            print(self.checkpoint)
            print("DEBUG::Categories:")
            print(self.encoder.categories)

        y = self.model.predict(X)
        # discard empty column edge case
        y[np.all(all_email_df.isnull(),axis=0)]=0

        NK_email_result = self.encoder.reverse_label_encode(y,p_threshold)
        print("Classification result is:")
        print(NK_email_result)

        elapsed_time = time.time()-start_time
        print("Total time for classification is : %.2f sec" % elapsed_time)
                
        if NK_email_result[0][0]: # empty edge case
            if DOMAIN_OBJECT=='attack':
                if NK_email_result[0][0][0] != 'friend':
                    result.prediction = 'true'
                    result.confidence = NK_email_result[1][0][0]
            else:
                result.prediction = NK_email_result[0][0][0]
                result.confidence = NK_email_result[1][0][0]

        return result


#-----
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grapevine_pb2_grpc.add_ClassifierServicer_to_server(NKEmailClassifier(), server)
    server.add_insecure_port('[::]:' + GRPC_PORT)
    server.start()
    restapp.run()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

@restapp.route("/healthcheck")
def health():
    return "HEALTHY"

if __name__ == '__main__':
    logging.basicConfig() # purpose?
    config = configparser.ConfigParser()
    config.read('config.ini')
    clfName = config['DEFAULT']['clfName']
    print("using clf name " + clfName + " ...")
    global CLF_NAME
    CLF_NAME = clfName
    modelName = config['DEFAULT']['modelName']
    print("using model " + modelName + " ...")
    global MODEL_OBJECT
    MODEL_OBJECT = modelName
    domain = config['DEFAULT']['domain']
    print("using domain " + domain + " ...")
    global DOMAIN_OBJECT
    DOMAIN_OBJECT = domain
    port_config = config['DEFAULT']['port_config']
    print("using port " + port_config + " ...")
    global GRPC_PORT
    GRPC_PORT = port_config
    
    serve()