#
# Test GRPC client code for NK Email Classifier
#
#

from __future__ import print_function
import logging

import grpc
import configparser
import grapevine_pb2
import grapevine_pb2_grpc

def run():

    channel = grpc.insecure_channel('localhost:' + GRPC_PORT)
    stub = grapevine_pb2_grpc.ClassifierStub(channel)

    testMessageHAM = grapevine_pb2.Message(
        raw="This raw field isn't used by keras spam classifier, only text field",
        text="The meeting has been rescheduled for next week sometime. I will send out an email providing some more details. This shouldn't affect the working group going forward, but I will also double check with the manager.",
    )

    testMessageSPAM = grapevine_pb2.Message(
        raw="This raw field isn't used...",
        text="Your email account has been hacked! Please log in to the following site to change your password.",
    )


    ### This should be classified as HAM
    classification = stub.Classify(testMessageHAM)
    confidence = classification.confidence
    print("Classifier gRPC client received this classifier score for HAM example (class/confidence): " + str(classification.prediction)+ "/" + str(confidence) + "\n\n")

    
    ### This should be classified as SPAM
    classification = stub.Classify(testMessageSPAM)
    confidence = classification.confidence
    print("Classifier gRPC client received this classifier score for SPAM example (class/confidence): " + str(classification.prediction) + "/" + str(confidence) + "\n\n")    




if __name__ == '__main__':
    logging.basicConfig() # purpose?
    config = configparser.ConfigParser()
    config.read('config.ini')
    port_config = config['DEFAULT']['port_config']
    print("using port " + port_config + " ...")
    global GRPC_PORT
    GRPC_PORT = port_config
    run()