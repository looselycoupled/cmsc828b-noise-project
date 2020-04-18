from tensor2tensor.utils.trainer_lib import create_hparams
from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment
from problem import *


PROBLEM = 'translate_cmsc828'
MODEL = 'transformer'
HPARAMS = 'transformer_base'

TRAIN_DIR = './translator/model_files'
DATA_DIR = './translator/'


# Initi Run COnfig for Model Training
RUN_CONFIG = create_run_config(
    model_dir=TRAIN_DIR, # Location of where model file is stored
    model_name=MODEL,

    # More Params here in this fucntion for controling how often to save checkpoints and more.
)


# Init Hparams object from T2T Problem
hparams = create_hparams(HPARAMS)
hparams.batch_size = 1024


# Create Tensorflow Experiment Object
tensorflow_exp_fn = create_experiment(
    run_config=RUN_CONFIG,
    hparams=hparams,
    model_name=MODEL,
    problem_name=PROBLEM,
    data_dir=DATA_DIR,
    train_steps=40, # Total number of train steps for all Epochs
    eval_steps=100 # Number of steps to perform for each evaluation
)

# Kick off Training
tensorflow_exp_fn.train_and_evaluate()