# Overview

This project extends Khayrallah et al's original work on the impact of various types of noise.  Our exploration attempts to train a transformer under the same conditions as the reference paper to see how a different model is impacted.

> Khayrallah, Huda, and Philipp Koehn. “On the Impact of Various Types of Noise on Neural Machine Translation.” Proceedings of the 2nd Workshop on Neural Machine Translation and Generation, Association for Computational Linguistics, 2018, pp. 74–83. ACLWeb, doi:10.18653/v1/W18-2709.

For our final experiments, we created a Docker image that was able to incorporate Tensor2Tensor in order to train models at different noise levels within a Kubernetes environment.  As such, it is best if you use Docker or Kubernetes on a cluster with Nvidia 2080 Ti (as the code assumes 10GB RAM available in the GPU).

For a local test run without a GPU see the Local Evaluation section below.

## Repository Organization

This repository contains the code for experimenting with several transformer libraries, as well as log files, kubernetes manifests, docker images, etc. that were used in final evaluation.  See below for more detail.

* `./cache/` : directory containing files small enough to be cached in the docker image including tarballs of the dev and evaluation source/target files.
* `./data/` : directory containing uncompressed versions of datasets used for testing purposes.
* `./docker/` : directory containing the Docker image definition and primary experiment file for the Tensor2Tensor training and evaluation jobs.
* `./harvardnlp/` : contains experimental code based on Harvard SEAS code.  This code does work but was not used for final results.
* `./kubernetes/` : contains final kubernetes manifests used to execute our training jobs.
* `./kubernetes/archive/` : contains kubernetes manifests used to fine tune hyperparameters and test our process.
* `./logs/` : contains training logs for final experiments used in project report
* `./notebooks/` : contains Jupyter notebooks used to evaluate training code and create report plots.
* `./utilities/` : contains Python code that was used when evaluating different model libraries
* `./utils/` : contains Python code that was used for final training jobs with Tensor2Tensor

## Local Demonstration

For a quick, local evaluation a script (`local.sh`) has been provided to start a minimal training job that should work on a relatively powerful computer.  The script notably does not use a GPU, uses a smaller dataset, and trains for less epochs but is otherwise the same process used in our final results.

The only requirement is that [Docker](https://www.docker.com/) is installed.  The image contains all required libraries and any external resources that are needed (datasets) will be downloaded automatically from Amazon S3. The code for the script is shown below.

```bash
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

```




## Downloading the Datasets

The datasets used for training can be downloaded from S3 using the links below

* [http://cmsc828b.s3.amazonaws.com/datasets/untranslated.01.tar.gz](http://cmsc828b.s3.amazonaws.com/datasets/untranslated.01.tar.gz)
* [http://cmsc828b.s3.amazonaws.com/datasets/untranslated.02.tar.gz](http://cmsc828b.s3.amazonaws.com/datasets/untranslated.02.tar.gz)
* [http://cmsc828b.s3.amazonaws.com/datasets/untranslated.03.tar.gz](http://cmsc828b.s3.amazonaws.com/datasets/untranslated.03.tar.gz)
* [http://cmsc828b.s3.amazonaws.com/datasets/untranslated.04.tar.gz](http://cmsc828b.s3.amazonaws.com/datasets/untranslated.04.tar.gz)
* [http://cmsc828b.s3.amazonaws.com/datasets/untranslated.05.tar.gz](http://cmsc828b.s3.amazonaws.com/datasets/untranslated.05.tar.gz)
* [http://cmsc828b.s3.amazonaws.com/datasets/untranslated.10.tar.gz](http://cmsc828b.s3.amazonaws.com/datasets/untranslated.10.tar.gz)
* [http://cmsc828b.s3.amazonaws.com/datasets/untranslated.20.tar.gz](http://cmsc828b.s3.amazonaws.com/datasets/untranslated.20.tar.gz)
* [http://cmsc828b.s3.amazonaws.com/datasets/untranslated.50.tar.gz](http://cmsc828b.s3.amazonaws.com/datasets/untranslated.50.tar.gz)
* [http://cmsc828b.s3.amazonaws.com/datasets/untranslated.100.tar.gz](http://cmsc828b.s3.amazonaws.com/datasets/untranslated.100.tar.gz)



## Creating the DataSets

To create the datasets from scratch, download the data files from the original paper at [http://data.statmt.org/noise/](http://data.statmt.org/noise/) and run the following commands to add untranslated text to the source and target files.

```bash
cat baseline.tok.de untranslated_de_trg.05.tok.de >source.05.de
cat baseline.tok.de untranslated_de_trg.10.tok.de >source.10.de
cat baseline.tok.de untranslated_de_trg.20.tok.de >source.20.de
cat baseline.tok.de untranslated_de_trg.50.tok.de >source.50.de
cat baseline.tok.de untranslated_de_trg.100.tok.de >source.100.de

cat baseline.tok.en untranslated_de_trg.05.tok.de >target.05.en
cat baseline.tok.en untranslated_de_trg.10.tok.de >target.10.en
cat baseline.tok.en untranslated_de_trg.20.tok.de >target.20.en
cat baseline.tok.en untranslated_de_trg.50.tok.de >target.50.en
cat baseline.tok.en untranslated_de_trg.100.tok.de >target.100.en
```

To clean html entities out of the resulting files use:

```bash
cat target.05.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.05.en
cat target.10.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.10.en
cat target.20.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.20.en
cat target.50.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.50.en
cat target.100.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.100.en

cat source.05.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.05.de
cat source.10.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.10.de
cat source.20.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.20.de
cat source.50.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.50.de
cat source.100.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.100.de
```

For convenience, below are the commands to tar the files up before putting on s3.

```bash
tar czvf untranslated.05.tar.gz source.05.de target.05.en
tar czvf untranslated.10.tar.gz source.10.de target.10.en
tar czvf untranslated.20.tar.gz source.20.de target.20.en
tar czvf untranslated.50.tar.gz source.50.de target.50.en
tar czvf untranslated.100.tar.gz source.100.de target.100.en
```

To create the new datasets from 1% to 4% and then take the html entities out run the following commands.

```bash
head -n 35200  untranslated_de_trg.100.tok.de >untranslated_de_trg.01.tok.de
head -n 70400  untranslated_de_trg.100.tok.de >untranslated_de_trg.02.tok.de
head -n 105600  untranslated_de_trg.100.tok.de >untranslated_de_trg.03.tok.de
head -n 140800  untranslated_de_trg.100.tok.de >untranslated_de_trg.04.tok.de

cat baseline.tok.de untranslated_de_trg.01.tok.de >source.01.de
cat baseline.tok.de untranslated_de_trg.02.tok.de >source.02.de
cat baseline.tok.de untranslated_de_trg.03.tok.de >source.03.de
cat baseline.tok.de untranslated_de_trg.04.tok.de >source.04.de

cat baseline.tok.en untranslated_de_trg.01.tok.de >target.01.en
cat baseline.tok.en untranslated_de_trg.02.tok.de >target.02.en
cat baseline.tok.en untranslated_de_trg.03.tok.de >target.03.en
cat baseline.tok.en untranslated_de_trg.04.tok.de >target.04.en

cat target.01.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.01.en
cat target.02.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.02.en
cat target.03.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.03.en
cat target.04.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.04.en

cat source.01.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.01.de
cat source.02.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.02.de
cat source.03.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.03.de
cat source.04.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.04.de

cd ../wrangled

tar czvf untranslated.01.tar.gz source.01.de target.01.en
tar czvf untranslated.02.tar.gz source.02.de target.02.en
tar czvf untranslated.03.tar.gz source.03.de target.03.en
tar czvf untranslated.04.tar.gz source.04.de target.04.en
```