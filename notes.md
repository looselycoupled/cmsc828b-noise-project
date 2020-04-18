asdf

```bash

export DATA_DIR="./data"
export TMP_DIR="./tmp"
export PROBLEM="translate_noise"
export USR_DIR="utils.problem"
export MODEL="transformer"
export HPARAMS="transformer_base"



t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM \
  --hparams_set= $HPARAMS\
  --t2t_usr_dir=$USR_DIR  # Path to a Python module that will be imported.

```



```bash
export TRAIN_DIR=tmp/model/
export batch_size=1024
export schedule=continuous_train_and_eval
export train_steps=100000
export eval_steps=1000

t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --hparams="batch_size=$batch_size" \
  --schedule=$schedule\
  --output_dir=$TRAIN_DIR \
  --train_steps=$train_steps \
  --eval_steps=$eval_steps \
  --t2t_usr_dir=$USR_DIR \
  --worker-gpu=1

```