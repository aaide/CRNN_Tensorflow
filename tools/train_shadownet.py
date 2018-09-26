#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午1:39
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : train_shadownet.py
# @IDE: PyCharm Community Edition
"""
Train shadow net script
"""
import os
import pickle
from typing import Tuple, Union

import tensorflow as tf
import os.path as ops
import time
import numpy as np
import argparse
from easydict import EasyDict
from hyperopt import hp, fmin, Trials, STATUS_FAIL, STATUS_OK, tpe, rand
from hyperopt.mongoexp import MongoTrials

from crnn_model import crnn_model
from local_utils import data_utils, log_utils
from local_utils.log_utils import compute_accuracy
from local_utils.config_utils import load_config

logger = log_utils.init_logger()


def init_args() -> Tuple[argparse.Namespace, EasyDict]:
    """
    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-dir', type=str,
                        help='Directory containing train_features.tfrecords')
    parser.add_argument('-c', '--chardict-dir', type=str,
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-m', '--model-dir', type=str,
                        help='Directory where to store model checkpoints')
    parser.add_argument('-t', '--tboard-dir', type=str,
                        help='Directory where to store TensorBoard logs')
    parser.add_argument('-f', '--config-file', type=str,
                        help='Use this global configuration file')
    parser.add_argument('-e', '--decode-outputs', action='store_true', default=False,
                        help='Activate decoding of predictions during training (slow!)')
    parser.add_argument('-w', '--weights-path', type=str, help='Path to pre-trained weights to continue training')
    parser.add_argument('-j', '--num-threads', type=int, default=int(os.cpu_count()/2),
                        help='Number of threads to use in batch shuffling')
    parser.add_argument('-x', '--max-hyper-evals', type=int, help="Number of hyperparameter evaluations for hyperopt")
    parser.add_argument('-o', '--mongo', type=str, help="Full URI to MongoDB instance to hold experiments."
                                                        " E.g. mongo://172.17.0.12:27017/simple-cnn")
    parser.add_argument('-k', '--exp-key', type=str, help="Key (identifier) to store this experiment in the MongoDB")

    args = parser.parse_args()

    config = load_config(args.config_file)
    if args.dataset_dir:
        config.cfg.PATH.TFRECORDS_DIR = args.dataset_dir
    if args.chardict_dir:
        config.cfg.PATH.CHAR_DICT_DIR = args.chardict_dir
    if args.model_dir:
        config.cfg.PATH.MODEL_SAVE_DIR = args.model_dir
    if args.tboard_dir:
        config.cfg.PATH.TBOARD_SAVE_DIR = args.tboard_dir
    if args.max_hyper_evals:
        config.cfg.HYPERTUNE.MAX_EVALS = args.max_hyper_evals
        config.cfg.HYPERTUNE.ENABLE = True
    if args.mongo:
        config.cfg.HYPERTUNE.MONGODB = args.mongo
        config.cfg.HYPERTUNE.ENABLE = True
    if args.exp_key:
        config.cfg.HYPERTUNE.EXP_KEY = args.exp_key
        config.cfg.HYPERTUNE.ENABLE = True

    return args, config.cfg


def create_objective(cfg: EasyDict, num_threads: int=2):
    """ Constructs an objective function for hyperopt. """

    def objective(params):
        cfg.ARCH.HIDDEN_UNITS = params['hidden_units']
        cfg.ARCH.HIDDEN_LAYERS = params['hidden_layers']
        cfg.TRAIN.BATCH_SIZE = params['batch_size']
        cfg.TRAIN.LEARNING_RATE = params['learning_rate']
        cfg.TRAIN.LR_DECAY_STEPS = params['lr_decay_steps']
        cfg.TRAIN.LR_DECAY_RATE = params['lr_decay_rate']
        cfg.TRAIN.LR_STAIRCASE = params['lr_staircase']
        cfg.TRAIN.MOMENTUM = params['momentum']

        try:
            out = train_shadownet(cfg, decode=False, num_threads=num_threads)
        except:
            return {'status': STATUS_FAIL, 'config': cfg}
        return {'status': STATUS_OK,
                'config': cfg,
                'loss': np.min(out)}  # TODO: check this

    return objective


def hyper_tune(cfg: EasyDict, space: dict, trials: Union[Trials, MongoTrials]) -> dict:
    """ Turn up the bass and get schwifty!

    :param cfg: config dict
    :param space:
    :param trials:

    """
    obj = create_objective(cfg)

    algo = {'tpe': tpe.suggest, 'random': rand.suggest}[cfg.HYPERTUNE.ALGORITHM]
    best = fmin(obj, space=space, algo=algo, trials=trials, max_evals=cfg.HYPERTUNE.MAX_EVALS)
    return best


def train_shadownet(cfg: EasyDict, weights_path: str=None, decode: bool=False, num_threads: int=4, save: bool=True) \
        -> np.array:
    """
    :param cfg: configuration EasyDict (e.g. global_config.config.cfg)
    :param weights_path: Path to stored weights
    :param decode: Whether to perform CTC decoding to report progress during training
    :param num_threads: Number of threads to use in tf.train.shuffle_batch
    :param save: Whether to save model checkpoints at each epoch
    :return History of values of the cost function
    """
    # decode the tf records to get the training data
    decoder = data_utils.TextFeatureIO(char_dict_path=ops.join(cfg.PATH.CHAR_DICT_DIR, 'char_dict.json'),
                                       ord_map_dict_path=ops.join(cfg.PATH.CHAR_DICT_DIR, 'ord_map.json')).reader

    input_images, input_labels, input_image_names = decoder.read_features(cfg, cfg.TRAIN.BATCH_SIZE, num_threads)

    shadownet = crnn_model.ShadowNet(phase='Train',
                                     hidden_nums=cfg.ARCH.HIDDEN_UNITS,
                                     layers_nums=cfg.ARCH.HIDDEN_LAYERS,
                                     num_classes=len(decoder.char_dict)+1)

    with tf.variable_scope('shadow', reuse=False):
        net_out = shadownet.build_shadownet(inputdata=input_images)

    cost = tf.reduce_mean(tf.nn.ctc_loss(labels=input_labels, inputs=net_out,
                                         sequence_length=cfg.ARCH.SEQ_LENGTH*np.ones(cfg.TRAIN.BATCH_SIZE)))

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(net_out,
                                                      cfg.ARCH.SEQ_LENGTH*np.ones(cfg.TRAIN.BATCH_SIZE),
                                                      merge_repeated=False)

    sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), input_labels))

    global_step = tf.Variable(0, name='global_step', trainable=False)

    starter_learning_rate = cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               cfg.TRAIN.LR_DECAY_STEPS, cfg.TRAIN.LR_DECAY_RATE,
                                               staircase=cfg.TRAIN.LR_STAIRCASE)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=cost, global_step=global_step)

    # Set tf summary
    os.makedirs(cfg.PATH.TBOARD_SAVE_DIR, exist_ok=True)
    tf.summary.scalar(name='Cost', tensor=cost)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    if decode:
        tf.summary.scalar(name='Seq_Dist', tensor=sequence_dist)
    merge_summary_op = tf.summary.merge_all()

    # Set saver configuration
    saver = tf.train.Saver()
    os.makedirs(cfg.PATH.TBOARD_SAVE_DIR, exist_ok=True)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(cfg.PATH.MODEL_SAVE_DIR, model_name)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = cfg.TRAIN.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(cfg.PATH.TBOARD_SAVE_DIR)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = cfg.TRAIN.EPOCHS

    with sess.as_default():
        if weights_path is None:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('Restore model from {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        patience_counter = 1
        cost_history = [np.inf]
        for epoch in range(train_epochs):
            if epoch > 1 and cfg.TRAIN.EARLY_STOPPING:
                # We always compare to the first point where cost didn't improve
                if cost_history[-1 - patience_counter] - cost_history[-1] > cfg.TRAIN.PATIENCE_DELTA:
                    patience_counter = 1
                else:
                    patience_counter += 1
                if patience_counter > cfg.TRAIN.PATIENCE_EPOCHS:
                    logger.info("Cost didn't improve beyond {:f} for {:d} epochs, stopping early.".
                                format(cfg.TRAIN.PATIENCE_DELTA, patience_counter))
                    break
            if decode:
                _, c, seq_distance, predictions, labels, summary = sess.run(
                    [optimizer, cost, sequence_dist, decoded, input_labels, merge_summary_op])

                labels = decoder.sparse_tensor_to_str(labels)
                predictions = decoder.sparse_tensor_to_str(predictions[0])
                accuracy = compute_accuracy(labels, predictions)

                if epoch % cfg.TRAIN.DISPLAY_STEP == 0:
                    logger.info('Epoch: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                        epoch + 1, c, seq_distance, accuracy))

            else:
                _, c, summary = sess.run([optimizer, cost, merge_summary_op])
                if epoch % cfg.TRAIN.DISPLAY_STEP == 0:
                    logger.info('Epoch: {:d} cost= {:9f}'.format(epoch + 1, c))

            cost_history.append(c)
            summary_writer.add_summary(summary=summary, global_step=epoch)
            if save:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

        return np.array(cost_history[1:])  # Don't return the first np.inf


if __name__ == '__main__':
    args, cfg = init_args()

    # Just a test
    search_space = {
        'hidden_units': hp.choice('hidden_units', [256, 512, 768, 1024]),
        'hidden_layers': hp.randint('hidden_layers', 2, 6),
        'batch_size': hp.choice('batch_size', [32, 64, 128, 256, 512]),
        'learning_rate': hp.loguniform('learning_rate', 0, 1) / 10.0,
        'lr_decay_steps': hp.choice('lr_decay_steps', [10, 20, 50, 100]),
        'lr_decay_rate': hp.loguniform('lr_decay_rate', 0, 1) / 10.0,
        'lr_staircase': hp.choice('lr_staircase', [False, True]),
        'momentum': 1 - hp.loguniform('momentum', 0, 1) / 10.0
        }

    if cfg.HYPERTUNE.ENABLE:
        if cfg.HYPERTUNE.MONGODB:
            assert args.exp_key, "Experiment key required for MongoTrials"

            cfg.PATH.MODEL_SAVE_DIR = os.path.join(cfg.PATH.MODEL_SAVE_DIR, args.exp_key)
            mongodb = args.mongo.strip("/") + "/jobs"
            trials = MongoTrials(mongodb, exp_key=args.exp_key)
            # This will block. Remember to start hyperopt-mongo-worker or use
            # tools/mongo-worker.py
            hyper_tune(cfg, search_space, trials)
        else:
            trials = Trials()
            hyper_tune(cfg, search_space, trials)
            with open(os.path.join(cfg.PATH.MODEL_SAVE_DIR, 'tpe_trials.p', 'wb')) as fd:
                pickle.dump(trials, fd, protocol=4)
    else:
        train_shadownet(cfg=cfg, weights_path=args.weights_path, decode=args.decode_outputs,
                        num_threads=args.num_threads)
    print('Done')
