# New Knowledge's spam classifier files

Assumptions in data labeling:

1. 419 emails are labeled as spam

2. enron emails are labeled as not spam

Built on top of the New Knowledge character-level convolutional neural network text classification system - SIMON

To setup:
`pip3.6 install -r requirements.txt`

To run, simply do
`python3.6 train_spam_classifier.py`

This assumes that you have already extracted enron and 419 emails using `preprocess-enron_kaggle.py` and 
`preprocess_nigerian_prince.py`, and pointed `train_spam_classifier.py` to the appropriate locations.

More documentation will be provided shortly...
