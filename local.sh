# use small dataset for demonstrataion.  See utils/problem.py for all dataset labels
export TRAIN_DATASET=mini

# limit train steps for demonstration purposes
export TRAIN_STEPS="500"

# config for model: transformer_big_single_gpu used for final experiments
export HPARAMS="transformer_base_v3"

export EXTRA_HPARAMS=""
export BATCH_SIZE="2048"
export DECODE_BATCH_SIZE="50"


docker run --env TRAIN_DATASET=$TRAIN_DATASET \
  --env TRAIN_STEPS=$TRAIN_STEPS \
  --env HPARAMS=$HPARAMS \
  --env EXTRA_HPARAMS=$EXTRA_HPARAMS \
  --env BATCH_SIZE=$BATCH_SIZE \
  --env DECODE_BATCH_SIZE=$DECODE_BATCH_SIZE \
  looselycoupled/cmsc828b-tensor2tensor ./execute.sh

