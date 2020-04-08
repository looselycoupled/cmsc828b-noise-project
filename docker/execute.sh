#!/bin/bash

function bail {
  echo "Subcommand failed"
  exit 1
}

# export TRAINING_FILE1=baseline.tok.de
# export TRAINING_FILE2=baseline.tok.en

curl http://cmsc828b.s3.amazonaws.com/data/$TRAINING_FILE1 >data/$TRAINING_FILE1
curl http://cmsc828b.s3.amazonaws.com/data/$TRAINING_FILE2 >data/$TRAINING_FILE2

python3 transformer-with-noise.py $TRAINING_FILE1 $TRAINING_FILE2