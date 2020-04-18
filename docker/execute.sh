export TRAIN_DIR="./checkpoints"
export DATA_DIR="./data"
export TMP_DIR="./cache"
export PROBLEM="translate_noise"
export USR_DIR="utils.problem"
export MODEL="transformer"
export HPARAMS="transformer_base_single_gpu"
export batch_size=4096
export schedule=continuous_train_and_eval
export train_steps=100000
export eval_steps=1000



t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM \
  --hparams_set= $HPARAMS\
  --t2t_usr_dir=$USR_DIR


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


pushd tmp/
tar -xzf test.tar.gz
popd

export DECODE_FILE="tmp/test/newstest2017.de"
BEAM_SIZE=4
ALPHA=0.6
batch_size=100

t2t-decoder \
  --t2t_usr_dir=$USR_DIR
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA,batch_size=$batch_size" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=translation.en

t2t-bleu --translation=translation.en --reference=tmp/test/newstest2017.en

tar -cvzf
