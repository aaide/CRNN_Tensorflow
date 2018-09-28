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
from hyperopt import fmin, Trials, STATUS_FAIL, STATUS_OK, tpe, rand
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
    parser.add_argument('-d', '--dataset_dir', type=str,
                        help='Directory containing train_features.tfrecords')
    parser.add_argument('-c', '--chardict_dir', type=str,
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-m', '--model_dir', type=str,
                        help='Directory where to store model checkpoints')
    parser.add_argument('-t', '--tboard_dir', type=str,
                        help='Directory where to store TensorBoard logs')
    parser.add_argument('-f', '--config_file', type=str,
                        help='Use this global configuration file')
    parser.add_argument('-e', '--decode_outputs', action='store_true', default=False,
                        help='Activate decoding of predictions during training (slow!)')
    parser.add_argument('-w', '--weights_path', type=str, help='Path to pre-trained weights to continue training')
    parser.add_argument('-j', '--num_threads', type=int, default=int(os.cpu_count()/2),
                        help='Number of threads to use in batch shuffling')

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

    return args, config.cfg


def create_objective(config: EasyDict, num_threads: int=2):
    """ Constructs an objective function for hyperopt. """

    def objective(params):

        # 1. update config with params

        # 2. Train

        try:
            out = train_shadownet(cfg, decode=False, num_threads=num_threads)
        except:
            return {'status': STATUS_FAIL, 'config': config}
        return {'status': STATUS_OK,
                'config': config,
                'loss': np.min(out)}  # TODO: check this

    return objective


def hyper_tune(configuration: EasyDict, space: dict, trials: Union[Trials, MongoTrials]) -> dict:
    """ Turn up the bass and get schwifty!

    :param configuration: config dict
    :param space:
    :param trials:

    """
    obj = create_objective(configuration)

    algo = {'tpe': tpe.suggest, 'random': rand.suggest}[configuration.HYPERTUNE.ALGORITHM]
    best = fmin(obj, space=space, algo=algo, trials=trials, max_evals=configuration.HYPERTUNE.MAX_EVALS)
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

    dataset = tf.data.TFRecordDataset(cfg.PATH.TFRECORDS_DIR)
    dataset = dataset.batch(cfg.TRAIN.BATCH_SIZE, drop_remainder=True)
    dataset = dataset.map(decoder.extract_features_batch, num_parallel_calls=num_threads)
    # dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=decoder.extract_features,
    #                                                       batch_size=config.cfg.TRAIN.BATCH_SIZE,
    #                                                       num_parallel_batches=num_threads,
    #                                                       drop_remainder=True))
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(cfg.TRAIN.BATCH_SIZE*num_threads*16))
    dataset = dataset.prefetch(buffer_size=cfg.TRAIN.BATCH_SIZE*num_threads)
    iterator = dataset.make_one_shot_iterator()
    input_images, input_labels, input_image_names = iterator.get_next()

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

    space = {}  # TODO

    if cfg.HYPERTUNE.ENABLE:
        if cfg.HYPERTUNE.MONGODB:
            assert args.exp_key, "Experiment key required for MongoTrials"

            cfg.PATH.MODEL_SAVE_DIR = os.path.join(cfg.PATH.MODEL_SAVE_DIR, args.exp_key)
            mongodb = args.mongo.strip("/") + "/jobs"
            trials = MongoTrials(mongodb, exp_key=args.exp_key)
            # This will block. Remember to start hyperopt-mongo-worker or use
            # tools/mongo-worker.py
            hyper_tune(cfg, space, trials)
        else:
            trials = Trials()
            hyper_tune(cfg, space, trials)
            with open(os.path.join(cfg.PATH.MODEL_SAVE_DIR, 'tpe_trials.p', 'wb')) as fd:
                pickle.dump(trials, fd, protocol=4)
    else:
        train_shadownet(cfg=cfg, weights_path=args.weights_path, decode=args.decode_outputs,
                        num_threads=args.num_threads)
    print('Done')
