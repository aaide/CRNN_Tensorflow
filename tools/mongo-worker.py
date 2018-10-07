#!/usr/bin/python3.5

# We need to import all modules required when running the jobs in the mongo db
import sys
import os
import uuid
import traceback
from tools.train_shadownet import train_shadownet
import pickle
from typing import Tuple, Union

import tensorflow as tf
import os.path as ops
import time
import numpy as np
import argparse
from easydict import EasyDict
from hyperopt import hp, fmin, Trials, STATUS_FAIL, STATUS_OK, tpe, rand
from hyperopt.mongoexp import MongoTrials, main_worker

from crnn_model import crnn_model
from local_utils import data_utils, log_utils
from local_utils.log_utils import compute_accuracy
from local_utils.config_utils import load_config

sys.exit(main_worker())
