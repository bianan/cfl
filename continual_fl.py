# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main classes and methods for simulating continual federated learning.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
from datetime import datetime
import json
import math
import os
import random
import sys
import concurrent.futures
import numpy as np
import tensorflow as tf
import linear_model
import utils
import word_lstm_model

tf.set_random_seed(99)

tf.app.flags.DEFINE_integer(
    'algorithm', 0, 'Indicates which algorithm to run.'
    'Options:'
    ' -1: baseline, runs FedAvg in each cycle.'
    '  0: retraining without model splitting.'
    '  1: algorithm 1 with model splitting.'
    '  2: algorithm 2 with model splitting.')
tf.app.flags.DEFINE_string('model_type', 'linear', 'Options: linear, lstm.')
tf.app.flags.DEFINE_boolean('verbose', False,
                     'Verbose mode will output some intermediate tensors.')
tf.app.flags.DEFINE_integer('num_clients', 2, 'Total Number of clients.')
tf.app.flags.DEFINE_integer('num_cycles', 2,
                     'Total numnber of cycles. At most 18 for LSTM.')
tf.app.flags.DEFINE_float('fraction_clients', 1,
                   'Fraction of clients randomly selected in FedAvg')
tf.app.flags.DEFINE_integer(
    'base_times', 1, 'Number of times to run one pass of all clients'
    'for FedAvg.')
tf.app.flags.DEFINE_boolean(
    'baseline_second_cycle_update', False,
    'Whether to update server from the beginning of the second cycle.')
tf.app.flags.DEFINE_integer('altmin_rounds', 2,
                     'Number of rounds of Alt Min in algorithm 2.')

FLAGS = tf.app.flags.FLAGS
FLAGS(sys.argv)

# Algorithm names, which are consistent with indices from FLAGS.algorithm.
ALGORITHM_NAMES = ('algorithm 0', 'algorithm 1', 'algorithm 2', 'baseline')

CLIENT_BASE_SCOPE = 'Client'
SERVER_SCOPE = 'Server'
# Parameters to control the federated training.
NUM_CLIENTS_PER_ROUND = int(FLAGS.fraction_clients * FLAGS.num_clients)
# Params for the continual setting.
PRINT_FREQUENCY = 4
num_rounds_fedavg_base = int(
    math.ceil(1.0 / FLAGS.fraction_clients) * FLAGS.base_times)

num_retrain_epochs_base = FLAGS.base_times
num_alternative_min_base = FLAGS.altmin_rounds * FLAGS.base_times

# for distinguish step summary and cycle summary
STEP_SUMM = 'step_summ'
CYCLE_SUMM = 'cycle_summ'


class AlgorithmConfig(object):
  """Configurations for the personalization algorithms."""

  class ConfigBaseline(object):
    num_epochs_per_round_fedavg = 1
    num_rounds_fedavg = num_rounds_fedavg_base
    # not directly used in the algorithm.
    num_retrain_epochs = num_retrain_epochs_base

  class ConfigAlgorithm0(object):
    num_epochs_per_round_fedavg = 1
    num_rounds_fedavg = num_rounds_fedavg_base
    num_retrain_epochs = num_retrain_epochs_base

  class ConfigAlgorithm1(object):
    num_epochs_per_round_fedavg = 1
    num_rounds_fedavg = num_rounds_fedavg_base
    num_retrain_epochs = num_retrain_epochs_base

  class ConfigAlgorithm2(object):
    num_epochs_per_round_fedavg = 1
    # used in the first cycle
    num_rounds_fedavg = num_rounds_fedavg_base
    num_retrain_epochs = num_retrain_epochs_base
    # used from the second cycle
    num_alternative_min = num_alternative_min_base
    num_rounds_fedavg_alter_min = num_rounds_fedavg_base
    num_retrain_epochs_alter_min = num_retrain_epochs_base

  config_collections = {
      -1: ConfigBaseline,
      0: ConfigAlgorithm0,
      1: ConfigAlgorithm1,
      2: ConfigAlgorithm2,
  }

  def __init__(self, algorithm_id):
    self.algorithm_config = self.config_collections[algorithm_id]()


class Agent(object):
  """Class for clients (id >= 0) and server (id == -1).

  Attributes:
    name: A unique string representing the name of the client, e.g., 'client_0'.
    id: A non-nonnegative integer, consistent with the name, e.g., it is 0 for a
      client named 'client_0'.
    model: An instance of the model class.
    update_ops_all: Update ops for all vars.
    update_ops_shared: Update ops for shared vars.
    dict_update_placeholders: A dict of var base name to its update-placeholder.
    read_ops_all_vars: Read ops for all vars.
    read_ops_shared_vars: Read ops for shared vars.

  Raises:
    ValueError: Unknown agent id.
  """

  def __init__(self,
               name,
               data_generator,
               model_class,
               configs=None,
               id_=-1,
               initializer=None):
    self.name = name
    self.id = id_
    self.data = data_generator(configs=configs, agent_id=id_)

    with tf.name_scope(utils.get_train_name_scope(name)):
      train_data = self.data.train_data_batch
      model_train = model_class(
          name,
          is_training=True,
          data=train_data,
          config=configs.train_config,
          initializer=initializer)

    with tf.name_scope(utils.get_validation_name_scope(name)):
      valid_data = self.data.validation_data_batch
      model_validation = model_class(
          name,
          is_training=False,
          data=valid_data,
          reuse=True,
          config=configs.train_config,
          initializer=initializer)

    with tf.name_scope(utils.get_test_name_scope(name)):
      test_data = self.data.test_data_batch
      model_test = model_class(
          name,
          is_training=False,
          data=test_data,
          reuse=True,
          config=configs.eval_config,
          initializer=initializer)

    self.model_train = model_train
    self.model_validation = model_validation
    self.model_test = model_test

    with tf.name_scope(utils.get_update_name_scope(self.name)):
      # One could use any of the three models in this update name scope, since
      # the vars are shared among them.
      update_ops_shared, placeholders_shared = utils.generate_update_ops(
          self.model_train.shared_vars)
      update_ops_personal, placeholders_personal = utils.generate_update_ops(
          self.model_train.personal_vars)
      update_ops_all = update_ops_shared + update_ops_personal
      # Merges two dicts of placeholders. placeholders_shared and
      # placeholders_personal should have no overlap keys.
      assert not set(placeholders_shared.keys()).intersection(
          placeholders_personal.keys())
      dict_update_placeholders = {}
      dict_update_placeholders.update(placeholders_shared)
      dict_update_placeholders.update(placeholders_personal)

    self.update_ops_all = update_ops_all
    self.update_ops_shared = update_ops_shared
    self.dict_update_placeholders = dict_update_placeholders

    self.read_ops_all_vars = {
        k: v.value() for k, v in self.model_train.var_dict.items()
    }
    self.read_ops_shared_vars = {
        utils.get_base_name(v): v.value() for v in self.model_train.shared_vars
    }

  def train(self,
            sess,
            num_epochs,
            update_vars_type=utils.VARS_TYPE_ALL,
            verbose=False):
    """Trains client for num_epochs.

    Args:
      sess: The TF Session.
      num_epochs: Number of epochs for training.
      update_vars_type: String. Options:
          utils.VARS_TYPE_ALL means all vars.
          utils.VARS_TYPE_SHARED means shared vars.
      verbose: Whether to output intermediate states of the training process.

    Raises:
      ValueError: Unknown update_vars_type.
      RuntimeError: When a server instance is being trained.
    """
    print('Training on client {} for {} epoch(s) ...'.format(
        self.id, num_epochs))
    for _ in range(num_epochs):
      self.train_one_epoch(sess, update_vars_type, verbose)

  def train_one_epoch(self,
                      sess,
                      update_vars_type=utils.VARS_TYPE_ALL,
                      verbose=False):
    """Trains client for one epoch.

    Args:
      sess: The TF Session.
      update_vars_type: String. Options:
          utils.VARS_TYPE_ALL means all vars.
          utils.VARS_TYPE_SHARED means shared vars.
      verbose: Whether prints training status or not.

    Raises:
      ValueError: Unknown update_vars_type.
      RuntimeError: When a server instance is being trained.
    """
    if self.id >= 0:
      self.model_train.run_one_epoch(sess, verbose, update_vars_type)
    else:
      raise RuntimeError('A server cannot be trained!')

  def get_validation_loss(self, sess):
    valid_loss = self.model_validation.run_one_epoch(sess, verbose=False)
    print('validation loss: {}'.format(valid_loss))
    return valid_loss

  def get_test_loss(self, sess):
    test_loss = self.model_test.run_one_epoch(sess, verbose=False)
    print('test loss: {}'.format(test_loss))
    return test_loss


def get_data_generator_and_model_class_and_configs():
  """Returns class names of data generator and model class."""
  if FLAGS.model_type == 'linear':
    return linear_model.RegressionData, linear_model.LinearRegression, linear_model.LinearRegressionConfig(
    )
  elif FLAGS.model_type == 'lstm':
    return (word_lstm_model.TextData, word_lstm_model.WordLSTM,
            word_lstm_model.LSTMConfig(FLAGS.config_type))
  else:
    raise ValueError('Unknown model type: %s' % FLAGS.model_type)


class Simulator(object):
  """Wraps clients, server and basic components for simulation."""

  def __init__(self, num_clients, data_generator, model_class, configs):
    self.num_clients = num_clients

    self.initializer = tf.random_uniform_initializer(
        -configs.train_config.init_scale, configs.train_config.init_scale)

    clients = {}
    for client_id in range(num_clients):
      name = CLIENT_BASE_SCOPE + '_%d' % client_id
      client = Agent(
          name,
          data_generator,
          model_class,
          configs=configs,
          id_=client_id,
          initializer=self.initializer)
      clients[client.name] = client
    self.clients = clients

    server_name = SERVER_SCOPE
    server = Agent(
        server_name, data_generator, model_class, configs=configs, id_=-1)
    self.server = server

    # Adds global step for writing summaries.
    self.global_step = tf.Variable(0, name='global_step')
    self.global_step_0 = tf.Variable(0, name='global_step_0')
    self.global_step_increment = self.global_step_0.assign_add(1)
    self.global_step_reset = tf.assign(self.global_step_0, 0)

    train_summary_scope = (
        CLIENT_BASE_SCOPE + '.*/' + utils.TRAIN_NAME + '.*/' +
        utils.LOSS_SUMMARY_NAME)
    valid_summary_scope = (
        CLIENT_BASE_SCOPE + '.*/' + utils.VALIDATION_NAME + '.*/' +
        utils.LOSS_SUMMARY_NAME)
    test_loss_scope = (
        CLIENT_BASE_SCOPE + '.*/' + utils.TEST_NAME + '.*/' +
        utils.LOSS_SUMMARY_NAME)

    self.train_summaries = tf.summary.merge_all(scope=train_summary_scope)
    self.valid_summaries = tf.summary.merge_all(scope=valid_summary_scope)
    self.test_summaries = tf.summary.merge_all(scope=test_loss_scope)

    # summary histograms
    self.train_perplexities_placeholder = tf.placeholder(tf.float32, [None])
    self.validation_perplexities_placeholder = tf.placeholder(
        tf.float32, [None])
    self.test_perplexities_placeholder = tf.placeholder(tf.float32, [None])

    self.train_perplexities_histogram = tf.summary.histogram(
        'perplexities_histogram/train', self.train_perplexities_placeholder)
    self.validation_perplexities_histogram = tf.summary.histogram(
        'perplexities_histogram/validation',
        self.validation_perplexities_placeholder)
    self.test_perplexities_histogram = tf.summary.histogram(
        'perplexities_histogram/test', self.test_perplexities_placeholder)

    # key will be the id of client.
    # One record will be denoted as (step, cycle_id, perplexity)
    self.train_log = collections.defaultdict(list)
    self.validation_log = collections.defaultdict(list)
    self.test_log = collections.defaultdict(list)
    self.cycle_id = -1

    # Used to have a different logdir for each run
    self.logdir = None

  def initialize(self, sess):
    """Reset global step and determine the log directories."""
    # Resets global step to be 0.
    sess.run(self.global_step_reset)
    now = datetime.now()
    time_string = now.strftime('%Y%m%d-%H%M%S')
    if FLAGS.algorithm == 2:
      # Creates subfolders for algorithm 2 since it has the parameter
      # FLAGS.altmin_rounds
      self.step_logdir = os.path.join(
          FLAGS.output_path, STEP_SUMM, ALGORITHM_NAMES[FLAGS.algorithm],
          '{}_altmin_rounds'.format(FLAGS.altmin_rounds), time_string)
      self.cycle_logdir = os.path.join(
          FLAGS.output_path, CYCLE_SUMM, ALGORITHM_NAMES[FLAGS.algorithm],
          '{}_altmin_rounds'.format(FLAGS.altmin_rounds), time_string)
    elif FLAGS.algorithm == -1:
      # Creates subfolders for baseline since it has the parameter
      # FLAGS.baseline_second_cycle_update
      self.step_logdir = os.path.join(
          FLAGS.output_path, STEP_SUMM, ALGORITHM_NAMES[FLAGS.algorithm],
          '2nd_cycle_update_{}'.format(FLAGS.baseline_second_cycle_update),
          time_string)
      self.cycle_logdir = os.path.join(
          FLAGS.output_path, CYCLE_SUMM, ALGORITHM_NAMES[FLAGS.algorithm],
          '2nd_cycle_update_{}'.format(FLAGS.baseline_second_cycle_update),
          time_string)
    else:
      self.step_logdir = os.path.join(FLAGS.output_path, STEP_SUMM,
                                      ALGORITHM_NAMES[FLAGS.algorithm],
                                      time_string)
      self.cycle_logdir = os.path.join(FLAGS.output_path, CYCLE_SUMM,
                                       ALGORITHM_NAMES[FLAGS.algorithm],
                                       time_string)

  def update_clients_from_server(self,
                                 sess,
                                 clients,
                                 update_vars_type=utils.VARS_TYPE_ALL):
    """Updates clients vars from server vars.

    Args:
      sess: TF Session.
      clients: A list of clients that will be updated from server.
      update_vars_type: String. Options: utils.VARS_TYPE_ALL means all vars,
        utils.VARS_TYPE_SHARED means shared vars.

    Raises:
      ValueError: Unknown update_vars_type.
    """
    if update_vars_type == utils.VARS_TYPE_ALL:
      server_vars = sess.run(self.server.read_ops_all_vars)
      client_update_ops = [c.update_ops_all for c in clients]

      client_update_ops_feed_dict = {}
      for c in clients:
        for var_base_name, placeholder in c.dict_update_placeholders.items():
          client_update_ops_feed_dict[placeholder] = np.array(
              [server_vars[var_base_name]])

    elif update_vars_type == utils.VARS_TYPE_SHARED:
      server_shared_vars = sess.run(self.server.read_ops_shared_vars)
      client_update_ops = [c.update_ops_shared for c in clients]
      client_update_ops_feed_dict = {}
      for c in clients:
        for shared_var in c.model_train.shared_vars:
          var_base_name = utils.get_base_name(shared_var)
          placeholder = c.dict_update_placeholders[var_base_name]
          client_update_ops_feed_dict[placeholder] = np.array(
              [server_shared_vars[var_base_name]])
    else:
      raise ValueError('Unknown vars update type: %s' % update_vars_type)

    sess.run(client_update_ops, feed_dict=client_update_ops_feed_dict)

  def update_server_from_clients(self,
                                 sess,
                                 clients,
                                 update_vars_type=utils.VARS_TYPE_ALL):
    """Updates server vars to be the weighted average of client vars.

    Args:
      sess: TF Session.
      clients: A list of clients that will be used to update server.
      update_vars_type: String. Options: utils.VARS_TYPE_ALL means all vars,
        utils.VARS_TYPE_SHARED means shared vars.

    Raises:
      ValueError: Unknown update_vars_type.
    """
    num_clients = len(clients)
    total_num_batches = 0
    for c in clients:
      total_num_batches += c.model_train.data.num_batches
    # client_weights should sum to num_clients.
    client_weights = [
        float(c.model_train.data.num_batches * num_clients / total_num_batches)
        for c in clients
    ]

    if update_vars_type == utils.VARS_TYPE_ALL:
      read_client_ops = collections.defaultdict(list)
      for var_base_name in self.server.model_train.var_dict:
        for c in clients:
          read_client_ops[var_base_name].append(
              c.read_ops_all_vars[var_base_name])

      client_vars = sess.run(read_client_ops)

      for cid, c in enumerate(clients):
        weight = client_weights[cid]
        for var_base_name in self.server.model_train.var_dict:
          client_vars[var_base_name][cid] *= weight

      server_feed_dict = {}
      for (var_base_name,
           placeholder) in self.server.dict_update_placeholders.items():
        client_vars_as_array = np.array(client_vars[var_base_name])
        server_feed_dict[placeholder] = client_vars_as_array

      sess.run(self.server.update_ops_all, feed_dict=server_feed_dict)

    elif update_vars_type == utils.VARS_TYPE_SHARED:
      read_client_ops = collections.defaultdict(list)
      for v in self.server.model_train.shared_vars:
        var_base_name = utils.get_base_name(v)
        for c in clients:
          read_client_ops[var_base_name].append(
              c.read_ops_shared_vars[var_base_name])
      client_vars = sess.run(read_client_ops)

      for cid, c in enumerate(clients):
        weight = client_weights[cid]
        for shared_var in self.server.model_train.shared_vars:
          var_base_name = utils.get_base_name(shared_var)
          client_vars[var_base_name][cid] *= weight

      server_feed_dict = {}
      for shared_var in self.server.model_train.shared_vars:
        var_base_name = utils.get_base_name(shared_var)
        client_vars_as_array = np.array(client_vars[var_base_name])
        placeholder = self.server.dict_update_placeholders[var_base_name]
        server_feed_dict[placeholder] = client_vars_as_array

      sess.run(self.server.update_ops_shared, feed_dict=server_feed_dict)

    else:
      raise ValueError('Unknown vars update type: %s' % update_vars_type)

  def fed_avg(self,
              sess,
              num_rounds=0,
              num_epochs_per_round=0,
              step_writer=None,
              update_vars_type=utils.VARS_TYPE_ALL):
    """Runs num_rounds of FedAvg.

    Args:
      sess: The TF Session.
      num_rounds: Number of rounds of FedAvg.
      num_epochs_per_round: Number of epochs in each round of FedAvg.
      step_writer: A tf.summary.FileWriter to write loss summaries every round.
      update_vars_type: String. utils.VARS_TYPE_ALL means all vars,
        utils.VARS_TYPE_SHARED means shared vars.

    Raises:
      ValueError: Unknown update_vars_type.
    """

    def _thread_fn(sess, client, num_epochs_per_round, update_vars_type):
      client.train(sess, num_epochs_per_round, update_vars_type, verbose=True)
      # Perplexity of the validation model will be updated automatically.
      client.get_validation_loss(sess)

    for r in range(num_rounds):
      # Picks a subset of clients uniformly.
      clients_for_this_round = random.sample(
          list(self.clients.values()), NUM_CLIENTS_PER_ROUND)

      # Updates client vars from server vars.
      self.update_clients_from_server(sess, clients_for_this_round,
                                      update_vars_type)
      if r % PRINT_FREQUENCY == 0 and FLAGS.verbose:
        print('Updated clients from server')

      threads = []
      executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=FLAGS.num_clients + 1)
      # Trains selected clients.
      for client in clients_for_this_round:
        threads.append(
            executor.submit(_thread_fn, sess, client, num_epochs_per_round,
                            update_vars_type))

      concurrent.futures.wait(threads)

      self.add_train_valid_summaries(sess, step_writer)

      if r % PRINT_FREQUENCY == 0 and FLAGS.verbose:
        print('Trained on %d clients for %d steps each' %
              (NUM_CLIENTS_PER_ROUND, num_epochs_per_round))

      # Updates server to be the average of selected clients.
      self.update_server_from_clients(sess, clients_for_this_round,
                                      update_vars_type)
      if r % PRINT_FREQUENCY == 0 and FLAGS.verbose:
        print('Updated server')

    # Adds test summary in the end of FedAvg.
    threads = []
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=FLAGS.num_clients + 1)
    for client in self.clients.values():
      threads.append(executor.submit(client.get_test_loss, sess))
      # client.get_test_loss(sess)
    concurrent.futures.wait(threads)
    self.add_test_summaries(sess, step_writer)

  def retrain_clients(self,
                      sess,
                      num_epochs,
                      step_writer,
                      update_vars_type=utils.VARS_TYPE_ALL,
                      verbose=False):
    """Retrains all vars or personal vars on all clients.

    Args:
      sess: The TF Session.
      num_epochs: Number of SGD steps for retrain.
      step_writer: A tf.summary.FileWriter to write loss summaries every epoch.
      update_vars_type: String. Options: utils.VARS_TYPE_ALL means all vars.
        utils.VARS_TYPE_PERSONAL means personal vars.
      verbose: Controls whether to output intermediate states of training.

    Raises:
      ValueError: Unknown update_vars_type.
    """

    def _thread_fn(sess, client, update_vars_type, verbose):
      print('Retraining {} for one epoch...'.format(client.name))
      client.train_one_epoch(sess, update_vars_type, verbose=verbose)
      client.get_validation_loss(sess)

    for _ in range(num_epochs):
      threads = []
      executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=FLAGS.num_clients + 2)
      # Have to sync each epoch in order to write summary.
      for client in self.clients.values():
        threads.append(
            executor.submit(_thread_fn, sess, client, update_vars_type,
                            verbose))
      concurrent.futures.wait(threads)
      self.add_train_valid_summaries(sess, step_writer)

    # Adds test summary in the end.
    threads = []
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=FLAGS.num_clients + 1)
    for client in self.clients.values():
      threads.append(executor.submit(client.get_test_loss, sess))
    concurrent.futures.wait(threads)

    self.add_test_summaries(sess, step_writer)

  def add_train_valid_summaries(self, sess, step_writer):
    """Adds summaries from training and validation processes."""
    summary_ops = {}
    summary_ops['train_summaries'] = self.train_summaries
    summary_ops['valid_summaries'] = self.valid_summaries
    summary_ops['global_step_0'] = self.global_step_increment
    summary_ops['train_histogram'] = self.train_perplexities_histogram
    summary_ops['validation_histogram'] = self.validation_perplexities_histogram

    train_perplexities = [
        c.model_train.perplexity
        for c in self.clients.values()
        if (c.model_train.perplexity is not None)
    ]
    validation_perplexities = [
        c.model_validation.perplexity
        for c in self.clients.values()
        if (c.model_validation.perplexity is not None)
    ]
    feed_dict = {
        self.train_perplexities_placeholder: train_perplexities,
        self.validation_perplexities_placeholder: validation_perplexities
    }
    for client in self.clients.values():
      feed_dict_train = client.model_train.get_summary_feed_dict()
      feed_dict_validation = client.model_validation.get_summary_feed_dict()
      assert set(feed_dict_train.keys()).intersection(
          set(feed_dict_validation.keys())) == set()
      feed_dict.update(feed_dict_train)
      feed_dict.update(feed_dict_validation)

    run_out = sess.run(summary_ops, feed_dict=feed_dict)
    step = run_out['global_step_0'] - 1
    step_writer.add_summary(run_out['train_summaries'], global_step=step)
    step_writer.add_summary(run_out['valid_summaries'], global_step=step)
    step_writer.add_summary(run_out['train_histogram'], global_step=step)
    step_writer.add_summary(run_out['validation_histogram'], global_step=step)

    # adds train valid logs for all clients.
    for client in self.clients.values():
      self.train_log[client.id].append((step, self.cycle_id,
                                        client.model_train.perplexity))
      self.validation_log[client.id].append(
          (step, self.cycle_id, client.model_validation.perplexity))

  def add_test_summaries(self, sess, step_writer):
    """Adds summaries from training and validation processes."""
    summary_ops = {}
    summary_ops['test_summaries'] = self.test_summaries
    summary_ops['global_step_0'] = self.global_step_0
    summary_ops['test_histogram'] = self.test_perplexities_histogram

    test_perplexities = [
        c.model_test.perplexity
        for c in self.clients.values()
        if (c.model_test.perplexity is not None)
    ]

    feed_dict = {self.test_perplexities_placeholder: test_perplexities}
    for client in self.clients.values():
      feed_dict_test = client.model_test.get_summary_feed_dict()
      feed_dict.update(feed_dict_test)

    run_out = sess.run(summary_ops, feed_dict=feed_dict)
    step = run_out['global_step_0'] - 1
    step_writer.add_summary(run_out['test_summaries'], global_step=step)
    step_writer.add_summary(run_out['test_histogram'], global_step=step)

    # adds test logs for all clients.
    for client in self.clients.values():
      self.test_log[client.id].append((step, self.cycle_id,
                                       client.model_test.perplexity))

  def add_cycle_summaries(self, sess, writer):
    """Adds summaries from training and validation processes."""
    summary_ops = {}
    summary_ops['train_summaries'] = self.train_summaries
    summary_ops['valid_summaries'] = self.valid_summaries
    summary_ops['train_histogram'] = self.train_perplexities_histogram
    summary_ops['validation_histogram'] = self.validation_perplexities_histogram
    summary_ops['test_summaries'] = self.test_summaries
    summary_ops['test_histogram'] = self.test_perplexities_histogram

    train_perplexities = [
        c.model_train.perplexity
        for c in self.clients.values()
        if (c.model_train.perplexity is not None)
    ]
    validation_perplexities = [
        c.model_validation.perplexity
        for c in self.clients.values()
        if (c.model_validation.perplexity is not None)
    ]
    test_perplexities = [
        c.model_test.perplexity
        for c in self.clients.values()
        if (c.model_test.perplexity is not None)
    ]
    feed_dict = {
        self.train_perplexities_placeholder: train_perplexities,
        self.validation_perplexities_placeholder: validation_perplexities,
        self.test_perplexities_placeholder: test_perplexities
    }
    for client in self.clients.values():
      feed_dict_train = client.model_train.get_summary_feed_dict()
      feed_dict_validation = client.model_validation.get_summary_feed_dict()
      feed_dict_test = client.model_test.get_summary_feed_dict()
      feed_dict.update(feed_dict_train)
      feed_dict.update(feed_dict_validation)
      feed_dict.update(feed_dict_test)

    run_out = sess.run(summary_ops, feed_dict=feed_dict)
    step = self.cycle_id
    writer.add_summary(run_out['train_summaries'], global_step=step)
    writer.add_summary(run_out['valid_summaries'], global_step=step)
    writer.add_summary(run_out['test_summaries'], global_step=step)
    writer.add_summary(run_out['train_histogram'], global_step=step)
    writer.add_summary(run_out['validation_histogram'], global_step=step)
    writer.add_summary(run_out['test_histogram'], global_step=step)

  def load_clients_data(self, cycle_id):
    """Reads running words in a cycle."""
    threads = []
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=FLAGS.num_clients + 1)
    for client in self.clients.values():
      threads.append(executor.submit(client.data.load_cycle_data, cycle_id))
    concurrent.futures.wait(threads)


class Experiment(object):
  """Class to do the Federated personalization experiments."""

  def __init__(self):
    (data_generator, model_class,
     configs) = get_data_generator_and_model_class_and_configs()

    self.simulator = Simulator(
        FLAGS.num_clients, data_generator, model_class, configs=configs)
    print('All clients: ', list(self.simulator.clients.keys()))
    self.algorithm_config = AlgorithmConfig(FLAGS.algorithm).algorithm_config
    self.log_folder_subfix = 'base_times_{}'.format(FLAGS.base_times)

  def write_log(self, logdir, logname):
    """Writes json logs to disk."""
    json_vars = {
        'num_clients': FLAGS.num_clients,
        'num_cycles': FLAGS.num_cycles,
        'fraction_clients': FLAGS.fraction_clients,
        'train': self.simulator.train_log,
        'validation': self.simulator.validation_log,
        'test': self.simulator.test_log,
        'algorithm_config': utils.get_attribute_dict(self.algorithm_config),
    }

    if not tf.gfile.IsDirectory(logdir):
      tf.gfile.MakeDirs(logdir)
    with tf.gfile.GFile(os.path.join(logdir, logname), 'w') as f:
      json.dump(json_vars, f, indent=2)

  def baseline(self):
    """Runs FedAvg in each cycle."""
    print('\n' + '=' * 10 + 'Running baseline' + '=' * 10 + '\n')

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=None,
        save_summaries_steps=None,
        save_summaries_secs=None) as sess:
      self.simulator.initialize(sess)
      step_writer = tf.summary.FileWriter(
          self.simulator.step_logdir, graph=sess.graph)
      cycle_writer = tf.summary.FileWriter(
          self.simulator.cycle_logdir, graph=sess.graph)

      for cycle in range(FLAGS.num_cycles):
        try:
          print('\n******Cycle %d starts\n' % cycle)
          self.simulator.cycle_id = cycle

          if cycle > 0:
            self.simulator.load_clients_data(cycle)
            if FLAGS.baseline_second_cycle_update:
              self.simulator.update_server_from_clients(
                  sess,
                  list(self.simulator.clients.values()),
                  update_vars_type=utils.VARS_TYPE_ALL)

          self.simulator.fed_avg(
              sess,
              self.algorithm_config.num_rounds_fedavg,
              num_epochs_per_round=self.algorithm_config
              .num_epochs_per_round_fedavg,
              step_writer=step_writer,
              update_vars_type=utils.VARS_TYPE_ALL)

          print('=========\n{} rounds of FedAvg done.\n'.format(
              self.algorithm_config.num_rounds_fedavg))
          step_writer.flush()
          # One cycle done.

        # Writes log every cycle.
        finally:
          self.write_log(self.simulator.step_logdir, logname='baseline.json')
          self.simulator.add_cycle_summaries(sess, cycle_writer)
          cycle_writer.flush()

      step_writer.close()
      cycle_writer.close()

    print('\n---Log is written to %s.' % self.simulator.step_logdir)

  def algorithm_0(self):
    """Runs FedAvg and Retrain without model splitting."""
    print('\n' + '=' * 10 + 'Running algorithm 0' + '=' * 10 + '\n')

    # Both save_summaries_steps and save_summaries_secs are set to None, in
    # order to disable the default summary saver of MonitoredTrainingSession,
    # since we want to control when to write summary using self-defined
    # summary step_writer.
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=None,
        save_summaries_steps=None,
        save_summaries_secs=None) as sess:
      self.simulator.initialize(sess)
      step_writer = tf.summary.FileWriter(
          self.simulator.step_logdir, graph=sess.graph)
      cycle_writer = tf.summary.FileWriter(
          self.simulator.cycle_logdir, graph=sess.graph)

      for cycle in range(FLAGS.num_cycles):
        try:
          print('\n******Cycle %d starts\n' % cycle)
          self.simulator.cycle_id = cycle

          if cycle > 0:
            self.simulator.load_clients_data(cycle)
            self.simulator.update_server_from_clients(
                sess,
                list(self.simulator.clients.values()),
                update_vars_type=utils.VARS_TYPE_ALL)

          self.simulator.fed_avg(
              sess,
              self.algorithm_config.num_rounds_fedavg,
              num_epochs_per_round=self.algorithm_config
              .num_epochs_per_round_fedavg,
              step_writer=step_writer,
              update_vars_type=utils.VARS_TYPE_ALL)

          print('=========\n{} rounds of FedAvg done.\n'.format(
              self.algorithm_config.num_rounds_fedavg))

          print('Retraining %d clients for %d epochs' %
                (FLAGS.num_clients, self.algorithm_config.num_retrain_epochs))

          self.simulator.retrain_clients(
              sess,
              self.algorithm_config.num_retrain_epochs,
              step_writer=step_writer,
              update_vars_type=utils.VARS_TYPE_ALL,
              verbose=True)
          step_writer.flush()
          # One cycle done.

        # Writes log every cycle.
        finally:
          self.write_log(self.simulator.step_logdir, logname='algorithm_0.json')
          self.simulator.add_cycle_summaries(sess, cycle_writer)
          cycle_writer.flush()

      step_writer.close()
      cycle_writer.close()

    print('\n---Step log is written to {}.'.format(self.simulator.step_logdir))

  def algorithm_1_model_splitting(self):
    """Runs algorithm 1 with model splitting."""
    print('\n' + '=' * 10 + 'Running Algorithm 1' + '=' * 10 + '\n')

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=None,
        save_summaries_steps=None,
        save_summaries_secs=None) as sess:
      self.simulator.initialize(sess)
      step_writer = tf.summary.FileWriter(
          self.simulator.step_logdir, graph=sess.graph)
      cycle_writer = tf.summary.FileWriter(
          self.simulator.cycle_logdir, graph=sess.graph)

      for cycle in range(FLAGS.num_cycles):
        try:
          print('\n******Cycle %d starts\n' % cycle)
          self.simulator.cycle_id = cycle

          # From cycle 1, initialize server to be the average of all clients.
          if cycle > 0:
            self.simulator.load_clients_data(cycle)
            self.simulator.update_server_from_clients(
                sess,
                list(self.simulator.clients.values()),
                update_vars_type=utils.VARS_TYPE_ALL)

          # print('\n=========\nServer vars:')
          # print(sess.run(self.simulator.server.read_ops_all_vars))

          # Run NUM_ROUNDS_FEDAVG of FedAvg.
          self.simulator.fed_avg(
              sess,
              self.algorithm_config.num_rounds_fedavg,
              num_epochs_per_round=self.algorithm_config
              .num_epochs_per_round_fedavg,
              step_writer=step_writer,
              update_vars_type=utils.VARS_TYPE_ALL)

          # Each client downloads the server model.
          self.simulator.update_clients_from_server(
              sess,
              list(self.simulator.clients.values()),
              update_vars_type=utils.VARS_TYPE_ALL)

          if FLAGS.verbose:
            print('After downloading the server model:')
            utils.print_vars_on_clients(self.simulator.clients, sess)

          print('Retraining personal vars of %d clients for %d epochs' %
                (FLAGS.num_clients, self.algorithm_config.num_retrain_epochs))
          self.simulator.retrain_clients(
              sess,
              self.algorithm_config.num_retrain_epochs,
              step_writer=step_writer,
              update_vars_type=utils.VARS_TYPE_PERSONAL,
              verbose=True)
          step_writer.flush()
        finally:
          self.write_log(self.simulator.step_logdir, logname='algorithm_1.json')
          self.simulator.add_cycle_summaries(sess, cycle_writer)
          cycle_writer.flush()

      step_writer.close()
      cycle_writer.close()
    print('\n---Log is written to {}.'.format(self.simulator.step_logdir))

  def algorithm_2_model_splitting(self):
    """Runs algorithm 2 with model splitting."""
    print('\n' + '=' * 10 + 'Algorithm 2 starts' + '=' * 10 + '\n')

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=None,
        save_summaries_steps=None,
        save_summaries_secs=None) as sess:
      self.simulator.initialize(sess)
      step_writer = tf.summary.FileWriter(
          self.simulator.step_logdir, graph=sess.graph)
      cycle_writer = tf.summary.FileWriter(
          self.simulator.cycle_logdir, graph=sess.graph)

      for cycle in range(FLAGS.num_cycles):
        try:
          self.simulator.cycle_id = cycle
          print('\n******Cycle %d starts\n' % cycle)
          if cycle == 0:
            # Run NUM_ROUNDS_FEDAVG of FedAvg.
            self.simulator.fed_avg(
                sess,
                self.algorithm_config.num_rounds_fedavg,
                num_epochs_per_round=self.algorithm_config
                .num_epochs_per_round_fedavg,
                step_writer=step_writer,
                update_vars_type=utils.VARS_TYPE_ALL)

            # Each client downloads the server model, and do splitting.
            # Here all clients need to participate.
            self.simulator.update_clients_from_server(
                sess,
                list(self.simulator.clients.values()),
                update_vars_type=utils.VARS_TYPE_ALL)

            if FLAGS.verbose:
              print('After downloading the server model, vars on clients are:')
              utils.print_vars_on_clients(self.simulator.clients, sess)

            print('Retraining personal vars of %d clients for %d epochs' %
                  (FLAGS.num_clients, self.algorithm_config.num_retrain_epochs))
            self.simulator.retrain_clients(
                sess,
                self.algorithm_config.num_retrain_epochs,
                step_writer=step_writer,
                update_vars_type=utils.VARS_TYPE_PERSONAL,
                verbose=True)

            print('Retrained on %d clients for %d steps.' %
                  (FLAGS.num_clients, self.algorithm_config.num_retrain_epochs))
            # utils.print_vars_on_clients(self.simulator.clients, sess)

          else:
            # For cycle 1, load new data and fetch average from clients.
            self.simulator.load_clients_data(cycle)
            self.simulator.update_server_from_clients(
                sess,
                list(self.simulator.clients.values()),
                update_vars_type=utils.VARS_TYPE_SHARED)

            # Starting from cycle 1, do AlternativeMin of shared vars
            # and personal vars.
            for round_alternative_min in range(
                self.algorithm_config.num_alternative_min):
              print('\n' + '-' * 3 +
                    'alternative_min round %d' % round_alternative_min)

              # Phase 1: Run few rounds of FedAvg on shared vars.
              self.simulator.fed_avg(
                  sess,
                  self.algorithm_config.num_rounds_fedavg_alter_min,
                  num_epochs_per_round=self.algorithm_config
                  .num_epochs_per_round_fedavg,
                  step_writer=step_writer,
                  update_vars_type=utils.VARS_TYPE_SHARED)

              # Phase 2: Retrain on personal vars.
              print('\n=====\nStart retraining personal vars on all clients.')
              self.simulator.retrain_clients(
                  sess,
                  self.algorithm_config.num_retrain_epochs_alter_min,
                  step_writer=step_writer,
                  update_vars_type=utils.VARS_TYPE_PERSONAL,
                  verbose=True)

              print('Trained on %d clients for %d steps.' %
                    (FLAGS.num_clients,
                     self.algorithm_config.num_retrain_epochs_alter_min))
            step_writer.flush()
            # Writes log every cycle.
        finally:
          self.write_log(self.simulator.step_logdir, logname='algorithm_2.json')
          self.simulator.add_cycle_summaries(sess, cycle_writer)
          cycle_writer.flush()

      step_writer.close()
      cycle_writer.close()
    print('\n---Log is written to {}.'.format(self.simulator.step_logdir))


def main(_):

  tf.reset_default_graph()
  experiment = Experiment()

  if FLAGS.output_path and FLAGS.config_type:
    FLAGS.output_path = os.path.join(FLAGS.output_path, FLAGS.config_type,
                                     experiment.log_folder_subfix)
  if FLAGS.algorithm == -1:
    experiment.baseline()
  elif FLAGS.algorithm == 0:
    experiment.algorithm_0()
  elif FLAGS.algorithm == 1:
    experiment.algorithm_1_model_splitting()
  elif FLAGS.algorithm == 2:
    experiment.algorithm_2_model_splitting()
  else:
    raise ValueError('Unknown algorithm id: %d!' % FLAGS.algorithm)


if __name__ == '__main__':
  tf.app.run()
