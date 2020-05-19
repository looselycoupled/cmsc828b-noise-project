#!/bin/bash

export TRAIN_DIR="./workspace/checkpoints"
export DATA_DIR="./workspace/data"
export TMP_DIR="./cache"
export PROBLEM="translate_noise"
export USR_DIR="utils.problem"
export MODEL="transformer"
export HPARAMS=${HPARAMS:-"transformer_base_single_gpu"}
export EXTRA_HPARAMS=${EXTRA_HPARAMS:-""}
export BATCH_SIZE=${BATCH_SIZE:-2048}
export schedule=continuous_train_and_eval
export TRAIN_STEPS=${TRAIN_STEPS:-100000}
export eval_steps=1000

export DECODE_FILE="cache/newstest2017.de"
export BEAM_SIZE=4
export ALPHA=0.6
export DECODE_BATCH_SIZE=${DECODE_BATCH_SIZE:-50}

function bail {
  echo "Subcommand failed"
  exit 1
}

mkdir $TRAIN_DIR
mkdir $DATA_DIR

echo HPARAMS: $HPARAMS
echo BATCH_SIZE: $BATCH_SIZE
echo TRAIN_STEPS: $TRAIN_STEPS
echo DECODE_BATCH_SIZE: $DECODE_BATCH_SIZE

echo "starting data generation"
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM \
  --hparams_set= $HPARAMS\
  --t2t_usr_dir=$USR_DIR

echo "starting training"
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --hparams="batch_size=${BATCH_SIZE}${EXTRA_HPARAMS}" \
  --schedule=$schedule\
  --output_dir=$TRAIN_DIR \
  --train_steps=$TRAIN_STEPS \
  --eval_steps=$eval_steps \
  --t2t_usr_dir=$USR_DIR \
  --keep_checkpoint_max=10 \
  --worker-gpu=1 2>&1 | tee training.log


pushd cache/
tar -xzf test.tar.gz
popd


echo "starting translation of test set"
t2t-decoder \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA,batch_size=$DECODE_BATCH_SIZE" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=translation.en

echo "starting calculation of bleu scores"
t2t-bleu --translation=translation.en --reference=cache/newstest2017.en 2>&1 | tee bleu-report.txt

TARBALL_FILE="${TRAIN_DATASET}.${TRAIN_STEPS}.checkpoints.`date +"%Y%m%d-%H%M"`.tar.gz"

echo "creating $TARBALL_FILE"
tar -cvzf ./workspace/$TARBALL_FILE ./workspace/checkpoints translation.en training.log bleu-report.txt
echo "copying $TARBALL_FILE to s3"
aws s3 cp ./workspace/$TARBALL_FILE s3://cmsc828b/checkpoints/$TARBALL_FILE
echo done!
