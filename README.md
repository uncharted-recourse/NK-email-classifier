# New Knowledge's Semantic Email Classifier

Assumptions in data labeling:

1. 419 emails are labeled as spam

2. enron emails are labeled as not spam

3. All JPL data abuse dataset emails are treated as foe - exceptions are FalsePositive and Recon which were dropped (former due to self-explanatory reason, and latter due to "lack of full understanding" reasons). Unknown was dropped as well.

Built on top of the New Knowledge character-level convolutional neural network text classification system - SIMON:

https://github.com/NewKnowledge/simon

You can swap out the multiclass multilabel model (enabled by default) for the binary model by modifying `config.ini` as specified in `deployed_checkpoints/checkpoint_descriptions.txt`

# gRPC Dockerized Classifier

The gRPC interface consists of the following components:
*) `grapevine.proto` in `protos/` which generates `grapevine_pb2.py` and `grapevine_pb2_grpc.py` according to instructions in `protos/README.md` -- these have to be generated every time `grapevine.proto` is changed
*) `spam_clf_server.py` which is the main gRPC server, serving on port `50052` (configurable at the top of that file)
*) `spam_clf_client.py` which is an example script demonstrating how the main gRPC server can be accessed to classify emails 
 
To build corresponding docker image:
`sudo docker build -t nk-email-classifier:latest .`

To run docker image, simply do
`sudo docker run -it -p 50052:50052 nk-email-classifier:latest`

Finally, edit `spam_clf_client.py` with example email of interest for classification, and then run that script as
`python3 spam_clf_client.py`


# REST Dockerized Classifier

Comment out gRPC server command at the bottom of `Dockerfile` (which is set as default serving protocol), and uncomment the REST server command. 

To build docker image:
`sudo docker build -t nk-email-classifier:latest .`

To run docker image, simply do
`sudo docker run -it -p 5000:5000 nk-email-classifier:latest`

Finally, edit `clientRestScriptExample.py` to fetch jsonl email of interest, and then run that script as
`python3 clientRestScriptExample.py`


Batch classification capabilities will be added next, for both serving protocols.
