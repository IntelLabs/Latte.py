#! /bin/bash
mkdir -p data
cd data
wget -nc http://web.stanford.edu/~lenny/imagenet_metadata/train_metadata.txt
wget -nc http://web.stanford.edu/~lenny/imagenet_metadata/val_metadata.txt
wget -nc http://web.stanford.edu/~lenny/imagenet_metadata/test_metadata.txt
wget -nc http://web.stanford.edu/~lenny/imagenet_metadata/synset_words.txt
