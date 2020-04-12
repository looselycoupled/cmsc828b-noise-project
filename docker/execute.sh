#!/bin/bash

# copy tokenizer cache up to code level where TF expects them to be
cp data/english.subwords .
cp data/other.subwords .

# fetch datasets for training
curl http://cmsc828b.s3.amazonaws.com/$TRAINING_FILE1 >$TRAINING_FILE1
curl http://cmsc828b.s3.amazonaws.com/$TRAINING_FILE2 >$TRAINING_FILE2

# output file listing to logs for sanity
ls -lia data/

python3 train.py $TRAINING_FILE1 $TRAINING_FILE2
