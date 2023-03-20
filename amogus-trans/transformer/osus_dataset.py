import sys
import csv
import glob
import json
import logging
import random
import os
from enum import Enum
from typing import List, Optional, Union
import numpy as np
import argparse
import pprint

from tqdm import tqdm, trange

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
from torch.utils.data import DataLoader, SequentialSampler

from transformers import (
    AutoTokenizer,
)

logger = logging.getLogger(__name__)

class OSUSDataset(Dataset):
    """Com2Sense Dataset."""

    def __init__(self, examples, tokenizer,
                 max_seq_length=None):
        """
        Args:
            examples (list): input examples of type `DummyExample`.
            tokenizer (huggingface.tokenizer): tokenizer in used.
            max_seq_length (int): maximum length to truncate the input ids.
            seed (int): random seed.
        """

        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # self.pad_id = self.tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        # self.sep_id = self.tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):

        ##################################################
        # TODO: Please finish this function.
        # Note that `token_type_ids` may not exist from
        # the outputs of tokenizer for certain types of
        # models (e.g. RoBERTa), please take special care
        # of it with an if-else statement.
        example = self.examples[idx]

        batch_encoding = self.tokenizer(
            text=example,
            max_length=self.max_seq_length,
            truncation=True,
            add_special_tokens=True
        )
        
        input_ids = torch.Tensor(batch_encoding["input_ids"]).long()
        
        attention_mask = torch.Tensor(batch_encoding["attention_mask"]).long()
        if "token_type_ids" not in batch_encoding:
            token_type_ids = torch.zeros_like(input_ids)
        else:
            token_type_ids = torch.Tensor(batch_encoding["token_type_ids"]).long()

        return input_ids, attention_mask, token_type_ids