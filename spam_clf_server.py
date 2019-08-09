#
# GRPC Server for NK Email Classifier
# 
# Uses GRPC service config in protos/grapevine.proto
# 

import nltk.data
from flask import Flask, request

import time
import pandas as pd
import numpy as np
import configparser
import os

from Simon import *
from Simon.Encoder import *

import grpc
import logging
from utils.log_func import get_log_func
import grapevine_pb2
import grapevine_pb2_grpc
from concurrent import futures

import tensorflow as tf
import tensorflow.keras.backend as K

tf_session = tf.Session()
K.set_session(tf_session)

log_level = os.getenv("LOG_LEVEL", "WARNING")
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
log = get_log_func(__name__)

log("starting flask app", level="debug")
restapp = Flask(__name__)
restapp.logger.debug('debug')

# GLOBALS
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

def LoadJSONLEmails(N = 50000, datapath=None, maxlen = 200, max_cells = 100):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    start = time.time()
    with open(datapath) as data_file:
        data_JSONL_lines = data_file.readlines()[:N]
    log(f"DEBUG::the current email type being loaded: {datapath}", level='debug')
    emails = [json.loads(line) for line in data_JSONL_lines]
    log('Content Loading {} took {} seconds'.format(datapath, time.time() - start), level='debug')
    lines = []
    for email in emails:
        txt = ''
        for url in email['urls']:
            txt += url + '.\n'
        txt += email['subject'] + '.\n'
        txt += email['body']
        sentences = [sentence[:maxlen] for sentence in tokenizer.tokenize(txt)][:max_cells]
        if len(sentences) == 0:
            continue
        none_count = max_cells - len(sentences)
        sentences = np.pad(sentences, (0,none_count), mode='wrap')
        lines.append(sentences)
    log('Parsing Loading {} took {} seconds'.format(datapath, time.time() - start), level='debug')
    return np.transpose(lines)

def parse_ta3_attack_labels(csv_file, datapath, fill = ['gather_general_info']):

    # read in csv files containing subject and label information
    labels = pd.read_csv(csv_file)
    labels = labels[~labels['Motive'].isnull()]
    
    # create data structure linking subject to (multi) labels
    labels = labels.set_index('Scenario')
    labels['labels'] = [[val for val in lst if str(val) != 'nan'] for lst in labels.values]
    labels_dict = labels.to_dict()['labels']
    
    # dictionary translating csv labels to model annotations
    label_to_annotation = {'Install malware': 'install_malware', 
        'Acquire credentials': 'acquire_credentials', 
        'Acquire PII': 'acquire_pii', 
        'Build trust': 'build_trust', 
        'Gain access to social network': 'access_social_network', 
        'Gather general information': 'gather_general_info', 
        'Get money': 'get_money', 
        'Make appointment': 'access_social_network',
    }

    # compare subject line of json to data structure
    annotations = []
    for idx, line in enumerate(open(datapath).readlines()):
        subject = json.loads(line)['subject']
        found = False
        for key, value in labels_dict.items():
            if key in subject or subject in key:
                labels = [label_to_annotation[v] for v in value]
                annotations.append(labels)
                found = True
                break
        if not found:
            # log index, subject if nothing found in data structure
            # fill with default values
            annotations.append(fill)
    return annotations

def prepare_data(raw_data, header, datapaths, labels, maxlen = 200, max_cells = 100):
    if raw_data is None:
        data = np.column_stack([LoadJSONLEmails(datapath=p, maxlen = maxlen, max_cells = max_cells) for p in datapaths])
        raw_data = data
    else:
        data = np.column_stack([LoadJSONLEmails(datapath=p, maxlen = maxlen, max_cells = max_cells) for p in datapaths])
        raw_data = np.column_stack((raw_data, data))
    if type(labels[0]) != list:
        header.extend([labels] * data.shape[1])
    else:
        header.extend(labels)
    return raw_data, header

def store_ta3_attack_data(maxlen = 200, max_cells = 100):
    ta3_header = []
    ta3_raw_data = None
    may_attacks = ["ta3-attacks/ta3-may-campaign.jsonl"]
    june_attacks = ["ta3-attacks/ta3-june-campaign.jsonl"]
    july_attacks = ["ta3-attacks/ta3-july-campaign.jsonl"]
    if DOMAIN_OBJECT == 'attack_category':
        may_annotations = parse_ta3_attack_labels("ta3-attacks/May_Campaign.csv", may_attacks[0], fill = ['gather_general_info', 'install_malware'])
        june_annotations = parse_ta3_attack_labels("ta3-attacks/June_Campaign.csv", june_attacks[0])
        july_annotations = parse_ta3_attack_labels("ta3-attacks/July_Campaign.csv", july_attacks[0])
    else:
        may_annotations = june_annotations = july_annotations =  ['foe']
    ta3_raw_data, ta3_header = prepare_data(ta3_raw_data, ta3_header,may_attacks, may_annotations, maxlen = maxlen, max_cells = max_cells)
    ta3_raw_data, ta3_header = prepare_data(ta3_raw_data, ta3_header,june_attacks, june_annotations, maxlen = maxlen, max_cells = max_cells)
    ta3_raw_data, ta3_header = prepare_data(ta3_raw_data, ta3_header,july_attacks, july_annotations, maxlen = maxlen, max_cells = max_cells)
    return ta3_raw_data, ta3_header

#-----
class NKEmailClassifier(grapevine_pb2_grpc.ClassifierServicer):

    def __init__(self):

        self.maxlen = 200 # length of each sentence
        self.max_cells = 100 # maximum number of sentences per email
        self.p_threshold = 0.5 # classifier decision boundary 

        # parameters for frequency and length of transfer updates
        self.transfer_sample_size = 500 # number of messages on which to fine tune classifier
        self.transfer_epochs = 1

        checkpoint_dir = "deployed_checkpoints/"
        execution_config=MODEL_OBJECT

        # load specified execution configuration
        if execution_config is None:
            raise TypeError
        Classifier = Simon(encoder={}) # dummy text classifier
        config = Classifier.load_config(execution_config, checkpoint_dir)
        self.encoder = config['encoder']
        self.checkpoint = config['checkpoint']

        # generate transfer model for training_phase
        self.transfer_queue = None
        with tf_session.as_default():
            with tf_session.graph.as_default():
                self.model = Classifier.generate_transfer_model(self.maxlen, self.max_cells, len(self.encoder.categories), 
                        len(self.encoder.categories), self.checkpoint, checkpoint_dir, activation='sigmoid')
                Classifier.load_weights(self.checkpoint,config,self.model,checkpoint_dir)
                self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
                self.model._make_predict_function()

        # parse ta-3 attack training labels once in the beginning
        self.ta3_raw_data, self.ta3_header = store_ta3_attack_data(maxlen = self.maxlen, max_cells = self.max_cells)
        self.ta3_header_original = self.ta3_header.copy()

    # Main classify function
    def Classify(self, request, context):

        # init classifier result object
        result = grapevine_pb2.Classification(
            domain=DOMAIN_OBJECT,
            prediction='false',
            confidence=0.0,
            model=CLF_NAME,
            version="0.0.9",
            meta=grapevine_pb2.Meta(),
        )

        # get text from input message
        input_doc = ''
        for url in request.urls:
            input_doc+=url + '.\n'
        input_doc += request.text 

        # Exception cases
        if (len(input_doc.strip()) == 0) or (input_doc is None):
            return result

        # SIMON NK_email_classifier prediction code
        start_time = time.time()
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(input_doc)
        sentences = [elem[-self.maxlen:] for elem in sentences][:self.max_cells]
        sentences = np.pad(sentences, (0,self.max_cells - len(sentences)), mode='wrap')
        
        # fine tune last layers of model on batched messages during training phase
        if request.training_phase:
            if self.transfer_queue is None:
                self.transfer_queue = np.reshape(sentences, (self.max_cells, 1))
                self.ta3_header = self.ta3_header_original.copy()
            elif self.transfer_queue.shape[1] == self.transfer_sample_size:

                # do transfer training
                combined_data = np.column_stack((self.ta3_raw_data, self.transfer_queue))
                self.ta3_header.extend([['friend']] * self.transfer_sample_size)
                combined_data = np.char.lower(np.transpose(combined_data).astype('U'))
                X, y, class_weights = self.encoder.encode_data(combined_data,self.ta3_header, self.maxlen)

                # decrease batch size to prevent errors
                batch_size = 64
                while batch_size >= 1:
                    try:
                        with tf_session.as_default():
                            with tf_session.graph.as_default():
                                self.model.fit(X, y, batch_size=batch_size, epochs=self.transfer_epochs, class_weight = class_weights)
                        restapp.logger.debug(f'Model was fine-tuned successfully on {len(self.ta3_header)} samples for {self.transfer_epochs} epochs')
                        break
                    except Exception as e:
                        restapp.logger.debug(f'Caught exception: {e} with batch_size {batch_size}, trying batch size {batch_size // 2} ')
                        batch_size = batch_size // 2
                self.transfer_queue = None

            else:
                # add current message to transfer queue
                self.transfer_queue = np.column_stack((self.transfer_queue, sentences))

        # SIMON NK_email_classifier prediction code cont.
        raw_data = np.char.lower(np.reshape(sentences, (1, self.max_cells)).astype('U'))
        X = self.encoder.x_encode(raw_data,self.maxlen)
        with tf_session.as_default():
            with tf_session.graph.as_default():
                y = self.model.predict(X)
        
        # package and return results
        NK_email_result = self.encoder.reverse_label_encode(y,self.p_threshold)
        if NK_email_result[0][0]: # empty edge case
            if DOMAIN_OBJECT=='attack':
                if NK_email_result[0][0][0] != 'friend':
                    result.prediction = 'true'
                result.confidence = NK_email_result[1][0][0]
            else:
                max_index = np.argmax(NK_email_result[1][0])
                result.domain = NK_email_result[0][0][max_index]
                result.confidence = NK_email_result[1][0][max_index]
                result.prediction = 'true'
                if result.domain == 'friend':
                    result.domain = 'gather_general_info'
                    result.prediction = 'false'
            restapp.logger.debug(f'result.domain: {result.domain}')
            restapp.logger.debug(f'result.prediction: {result.prediction}')
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
    config = configparser.ConfigParser()
    config.read('config.ini')
    global CLF_NAME
    CLF_NAME = config['DEFAULT']['clfName']
    log(f"using clf name {CLF_NAME}", level='debug')
    global MODEL_OBJECT 
    MODEL_OBJECT = config['DEFAULT']['modelName']
    log(f"using model {MODEL_OBJECT}", level='debug')
    global DOMAIN_OBJECT
    DOMAIN_OBJECT = config['DEFAULT']['domain']
    log(f"using domain {DOMAIN_OBJECT}", level='debug')
    global GRPC_PORT
    GRPC_PORT = config['DEFAULT']['port_config']
    log(f"using port {GRPC_PORT}", level='debug')
    serve()