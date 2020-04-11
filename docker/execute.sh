#!/bin/bash

function bail {
  echo "Subcommand failed"
  exit 1
}

# export TRAINING_FILE1=baseline.tok.de
# export TRAINING_FILE2=baseline.tok.en

cp data/english.subwords .
cp data/other.subwords .

curl http://cmsc828b.s3.amazonaws.com/$TRAINING_FILE1 >$TRAINING_FILE1
curl http://cmsc828b.s3.amazonaws.com/$TRAINING_FILE2 >$TRAINING_FILE2

ls -lia data/

python3 transformer-with-noise.py $TRAINING_FILE1 $TRAINING_FILE2
