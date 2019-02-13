# New Knowledge's spam classifier files

Assumptions in data labeling:

1. 419 emails are labeled as spam

2. enron emails are labeled as not spam

3. All JPL data abuse dataset emails are treated as foe - exceptions are FalsePositive and Recon which were dropped (former due to self-explanatory, and latter due to "lack of full understanding" reasons). Unknown was dropped as well.

Built on top of the New Knowledge character-level convolutional neural network text classification system - SIMON:

https://github.com/NewKnowledge/simon


To build docker image:
`sudo docker build -t nk-email-classifier:latest .`

To run docker image, simply do
`sudo docker run -it -p 5000:5000 nk-email-classifier:latest`

Finally, edit `clientRestScriptExample.py` to fetch jsonl email of interest, and then run that scripts as
`python3 clientRestScriptExample.py`
