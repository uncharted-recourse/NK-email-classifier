# start from a pinned version of tensorflow gpu with python 3 on ubuntu 18.04
FROM tensorflow/tensorflow:1.14.0-gpu-py3

ENV HOME=/root
WORKDIR $HOME

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# update os package manager, then install prerequisite packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git

# install base requirements
COPY requirements.txt $HOME/
RUN pip install -r requirements.txt

# copy model files
COPY deployed_checkpoints/ $HOME/deployed_checkpoints

# copy attack training data
COPY ta3_attacks/ $HOME/ta3-attacks

# copy everything else (excluding stuff specified in .dockerignore)
COPY . $HOME/

# install NLTK data
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data punkt
   
# make a non-root user group and add a user
RUN groupadd -g 1001 appuser && \
    useradd -r -u 1001 -g appuser appuser

# give user group access to home directory
RUN chown 1001:1001 $HOME

RUN chmod +x start_gRPC.sh && \
    sync

USER appuser 

# this is the gRPC server command
#ENV FLASK_ENV=development
CMD ./start_gRPC.sh