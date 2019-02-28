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


# GLOBALS
GRPC_PORT = '50052'

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

# CLASS_MAP = {0: 'ham', 1: 'spam'} # purpose?

#-----
class NKEmailClassifier(grapevine_pb2_grpc.ClassifierServicer):

    # Main classify function
    def Classify(self, request, context):

        # init classifier result object
        result = grapevine_pb2.Classification(
            domain='attack',
            prediction='false',
            confidence=0.0,
            model="NK_email_classifer",
            version="0.0.1",
            meta=grapevine_pb2.Meta(),
        )

        # get text from input message
        input_doc = request.text

        # Exception cases
        if (len(input_doc.strip()) == 0) or (input_doc is None):
            return result

        # SIMON NK_email_classifier prediction code
        sample_email = input_doc
        start_time = time.time()
        # set important parameters
        maxlen = 200 # length of each sentence
        max_cells = 100 # maximum number of sentences per email
        p_threshold = 0.5 # decision boundary

        DEBUG = True # boolean to specify whether or not print DEBUG information

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        checkpoint_dir = "deployed_checkpoints/"
        
        print("DEBUG::sample email (whole, then tokenized into sentences):")
        print(sample_email)
        sample_email_sentence = tokenizer.tokenize(sample_email)
        sample_email_sentence = [elem[-maxlen:] for elem in sample_email_sentence] # truncate
        print(sample_email_sentence)
        all_email_df = pd.DataFrame(sample_email_sentence,columns=['Email 0'])
        
        all_email_df = all_email_df.astype(str)
        all_email_df = pd.DataFrame.from_records(DataLengthStandardizerRaw(all_email_df,max_cells))

        print("DEBUG::the final shape is:")
        print(all_email_df.shape)

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

        model = Classifier.generate_model(maxlen, max_cells, category_count,activation='softmax')
        Classifier.load_weights(checkpoint,config,model,checkpoint_dir)
        model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['binary_accuracy'])

        y = model.predict(X)
        # discard empty column edge case
        y[np.all(all_email_df.isnull(),axis=0)]=0

        NK_email_result = encoder.reverse_label_encode(y,p_threshold)
        print("Classification result is:")
        print(NK_email_result)

        elapsed_time = time.time()-start_time
        print("Total time for classification is : %.2f sec" % elapsed_time)
        
        Classifier.clear_session()  # critical for enabling repeated calls of function
        
        if NK_email_result[0][0]: # empty edge case
            if NK_email_result[0][0][0] != 'friend':
                result.prediction = 'true'
                result.confidence = NK_email_result[1][0][0]

        return result


#-----
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grapevine_pb2_grpc.add_ClassifierServicer_to_server(NKEmailClassifier(), server)
    server.add_insecure_port('[::]:' + GRPC_PORT)
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig() # purpose?
    config = configparser.ConfigParser()
    config.read('config.ini')
    modelName = config['DEFAULT']['modelName']
    print("using model " + modelName + " ...")
    global MODEL_OBJECT
    MODEL_OBJECT = modelName
    serve()