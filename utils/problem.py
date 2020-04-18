# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

DATASETS = {
    "mini": [
        [
            "http://cmsc828b.s3.amazonaws.com/datasets/mini.tar.gz",
            ("baseline.50000.tok.de", "baseline.50000.tok.de",)
        ],
    ],
    "baseline": [
        [
            "http://cmsc828b.s3.amazonaws.com/datasets/baseline.tar.gz",
            ("baseline.tok.de", "baseline.tok.en",)
        ],
    ],

    "eval": [
        [
            "http://cmsc828b.s3.amazonaws.com/datasets/eval.tar.gz",
            ("newstest2016.de", "newstest2016.en")
        ],
    ],
    "test": [
        [
            "http://cmsc828b.s3.amazonaws.com/datasets/test.tar.gz",
            ("newstest2017.de", "newstest2017.en")
        ],
    ]
}


@registry.register_problem
class TranslateNoise(translate.TranslateProblem):
    """de-en translation trained on europarl corpus."""

    @property
    def additional_training_datasets(self):
        """Allow subclasses to add training datasets."""
        return []

    def source_data_files(self, dataset_split):
        if dataset_split == problem.DatasetSplit.TRAIN:

            return DATASETS[os.environ["TRAIN_DATASET"]]

        if dataset_split == problem.DatasetSplit.EVAL:
            return DATASETS['eval']

        if dataset_split == problem.DatasetSplit.TEST:
            return DATASETS['test']

        raise Exception("unknown dataset_split")

    @property
    def is_generate_per_split(self):
        return True

