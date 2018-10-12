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
"""Word LSTM model and data interface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import json
import os
import time
import numpy as np
import tensorflow as tf
import utils


tf.app.flags.DEFINE_string("output_path", "/tmp/log/fl_simulation",
                    "Output directory.")
tf.app.flags.DEFINE_string("config_type", "debug",
                    "Options are: debug, small, medium, large.")
tf.app.flags.DEFINE_string(
    "data_path_reddit",
    "//tmp/data/bq-20",
    "Contains reddit comment dataset for all months.")
tf.app.flags.DEFINE_integer(
    "num_gpus", 0, "If larger than 1, Grappler AutoParallel optimizer "
    "will create multiple training replicas with each GPU "
    "running one replica.")
tf.app.flags.DEFINE_string(
    "rnn_mode", "basic", "The low level implementation of lstm cell: one of "
    "'basic', and 'block', "
    "representing basic_lstm, and lstm_block_cell classes.")
tf.app.flags.DEFINE_string("optimizer", "adam", "Options: sgd, adam")
tf.app.flags.DEFINE_integer(
    "model_split", 0, "Model splitting strategy."
    "0: embedding, hidden | softmax, 1: embedding | hidden softmax")
FLAGS = tf.app.flags.FLAGS


BASIC = "basic"
BLOCK = "block"
# Fractions of data used for training and validation, respectively.
# Fraction of test data is: 1 - TRAIN_FRACTION - VALID_FRACTION.
TRAIN_FRACTION = 0.8
VALID_FRACTION = 0.1
# Fraction of training data used in debug mode.
DEBUG_FRACTION_DATA = 0.02

MONTHS = ("2015_01", "2015_02", "2015_03", "2015_04", "2015_05", "2015_06",
          "2015_07", "2015_08", "2015_09", "2015_10", "2015_11", "2015_12",
          "2016_01", "2016_02", "2016_03", "2016_04", "2016_05", "2016_06")
# The "subreddits" selected from reddit comment dataset.
SUBREDDITS = ("science", "funny", "sports", "worldnews", "pics", "gaming",
              "videos", "movies", "Music", "blog", "gifs", "explainlikeimfive",
              "books", "television", "EarthPorn", "DIY", "food",
              "Documentaries", "history", "InternetIsBeautiful", "funny")


def export_state_tuples(state_tuples, name):
  for state_tuple in state_tuples:
    tf.add_to_collection(name, state_tuple.c)
    tf.add_to_collection(name, state_tuple.h)


def import_state_tuples(state_tuples, name, num_replicas):
  restored = []
  for i in range(len(state_tuples) * num_replicas):
    c = tf.get_collection_ref(name)[2 * i + 0]
    h = tf.get_collection_ref(name)[2 * i + 1]
    restored.append(tf.contrib.rnn.LSTMStateTuple(c, h))
  return tuple(restored)


class LSTMConfig(object):
  """Configurations for LSTM model."""

  class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1e-3
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    hidden_size = 100
    num_epochs_with_init_learning_rate = 4
    total_num_epochs = 13
    keep_prob = 1.0
    learning_rate_decay = 0.5
    batch_size = 20
    num_samples = 1000
    vocab_size = 10000
    rnn_mode = "basic"
    data_keep_fraction = 1.0
    embedding_size = 100
    adam_learning_rate = 1e-3

  class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1e-3
    max_grad_norm = 5
    num_layers = 1
    num_steps = 30
    hidden_size = 200
    num_epochs_with_init_learning_rate = 6
    total_num_epochs = 39
    keep_prob = 0.5
    learning_rate_decay = 0.8
    batch_size = 20
    num_samples = 1000
    vocab_size = 10000
    rnn_mode = "basic"
    data_keep_fraction = 1.0
    embedding_size = 200
    adam_learning_rate = 1e-3

  class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1e-3
    max_grad_norm = 10
    num_layers = 1
    num_steps = 35
    hidden_size = 600
    num_epochs_with_init_learning_rate = 14
    total_num_epochs = 55
    keep_prob = 0.35
    learning_rate_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
    num_samples = 1000
    rnn_mode = "basic"
    data_keep_fraction = 1.0
    embedding_size = 600
    adam_learning_rate = 1e-3

  class DebugConfig(object):
    """XSmall config, for debugging."""
    init_scale = 0.1
    learning_rate = 1e-3
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 3
    num_epochs_with_init_learning_rate = 1
    total_num_epochs = 1
    keep_prob = 1.0
    learning_rate_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    num_samples = 1000
    rnn_mode = "basic"
    data_keep_fraction = DEBUG_FRACTION_DATA
    embedding_size = 4
    adam_learning_rate = 1e-3

  config_collections = {
      "small": SmallConfig,
      "medium": MediumConfig,
      "large": LargeConfig,
      "debug": DebugConfig,
  }

  def __init__(self, config_type):
    # Firstly set eval_config to have the same batch_size and num_steps
    # as train_config. May try different settings later in experiments.
    self.train_config = self.config_collections[config_type]()
    self.eval_config = self.config_collections[config_type]()


class TextDataBatch(object):
  """Text data batch generator.

  Attributes:
    input: A tensor. One batch of data, with the shape of [batch_size,
      num_steps]. Each entry is a word id.
    target: A tensor. One batch of target data, with the same shape as input,
      but time-shifted to the right by one. Each entry is a word id.
    num_batches: Number of data block pairs (input, target) in one
      epoch.
  """

  def __init__(self, config, name=None):
    """Constructs one batch of data."""
    self.name = name
    self.batch_size = config.batch_size
    self.num_steps = config.num_steps

    self.input = tf.placeholder(
        dtype=tf.int32, shape=[self.batch_size, self.num_steps])
    self.target = tf.placeholder(
        dtype=tf.int32, shape=[self.batch_size, self.num_steps])

    # shape will be:  batch_size * num_total_steps
    self.batched_data = None

  def _generate_batched_data(self, raw_data):
    """Creates batched data from the raw_data."""
    # raw token ids.
    self.raw_data = raw_data
    self.data_len = len(raw_data)
    num_total_steps = self.data_len // self.batch_size
    batched_data = np.reshape(raw_data[0:self.batch_size * num_total_steps],
                              [self.batch_size, num_total_steps])

    self.num_batches = (num_total_steps - 1) // self.num_steps

    assert self.num_batches > 0, ("num_batches==0, decrease batch_size or "
                                  "num_steps")
    return batched_data

  def update_batched_data(self, new_raw_data):
    self.batched_data = self._generate_batched_data(new_raw_data)

  def fetch_a_batch(self, batch_id=0):
    """Fecthes a batch from self.batched_data."""
    if batch_id < 0 or batch_id > self.num_batches - 1:
      raise ValueError("batch_id is out of range.")

    data = self.batched_data
    batch_size = self.batch_size
    num_steps = self.num_steps
    starting_col_id = batch_id * num_steps
    x = data[0:batch_size, starting_col_id:starting_col_id + num_steps]
    y = data[0:batch_size, starting_col_id + 1:starting_col_id + num_steps + 1]
    return x, y

  def get_batch_feed_dict(self, batch_id):
    x, y = self.fetch_a_batch(batch_id)
    feed_dict = {self.input: x, self.target: y}
    return feed_dict


class TextData(object):
  """Text data generator.

  Attributes:
    agent_id: Specifies the index of subreddits.
  """

  def __init__(self,
               configs,
               data_keep_fraction=1.0,
               agent_id=0,
               cycle_id=0,
               name="Data"):
    """Constructs batch tensors for train, validation and test.

    Args:
      configs: An instance of LSTMConfig class.
      data_keep_fraction: If in debug mode, only use a small fraction of data.
      agent_id: id of the reddit user.
      cycle_id: Id of the episode.
      name: Name of the op.
    """
    self.agent_id = agent_id
    self.cycle_id = cycle_id
    self.data_path = FLAGS.data_path_reddit
    self.data_keep_fraction = data_keep_fraction

    self.train_data_batch = TextDataBatch(
        configs.train_config, name=utils.TRAIN_NAME + name)
    self.validation_data_batch = TextDataBatch(
        configs.train_config, name=utils.VALIDATION_NAME + name)
    self.test_data_batch = TextDataBatch(
        configs.eval_config, name=utils.TEST_NAME + name)
    self.load_cycle_data(self.cycle_id)

    configs.train_config.vocab_size = self.vocab_size
    configs.eval_config.vocab_size = self.vocab_size

  def load_cycle_data(self, cycle_id):
    """Loads the data in a cycle."""

    (self.train_data, self.validation_data,
     self.test_data, self.vocab_size) = self._read_raw_data(
         self.data_path, self.data_keep_fraction, cycle_id)
    print(
        "cycle {}, number of samples on agent {}".format(
            cycle_id, self.agent_id), len(self.train_data),
        len(self.validation_data), len(self.test_data))
    self.train_data_batch.update_batched_data(self.train_data)
    self.validation_data_batch.update_batched_data(self.validation_data)
    self.test_data_batch.update_batched_data(self.test_data)

  def _read_raw_data(self, data_path=None, data_keep_fraction=1.0, cycle_id=0):
    """Loads raw text data from data directory "data_path".

    Reads text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    Args:
      data_path: String path to the data directory.
      data_keep_fraction: Fraction of data to be kept.
      cycle_id: Id of the cycle.

    Returns:
      tuple (train_data, valid_data, test_data, vocabulary_size).

    Raises:
      ValueError: Unknown dataset name.
    """

    file_path = os.path.join(data_path, MONTHS[cycle_id],
                             MONTHS[cycle_id] + ".json")
    all_words = self._read_subreddits(file_path, SUBREDDITS[self.agent_id])
    data_length = len(all_words)
    all_words = all_words[:int(data_length * data_keep_fraction)]
    word_to_id = self._load_subreddits_vocab(data_path)

    # The vocab contains one out-of-vocab token, which captures all tokens that
    # are not in the vocab.
    vocabulary_size = len(word_to_id) + 1

    all_data = self._word_to_word_ids(all_words, word_to_id)
    length_all_data = len(all_data)

    num_train_words = int(length_all_data * TRAIN_FRACTION)
    num_valid_words = int(length_all_data * VALID_FRACTION)

    train_data = all_data[:num_train_words]
    valid_data = all_data[num_train_words:num_train_words + num_valid_words]
    test_data = all_data[num_train_words + num_valid_words:]

    return train_data, valid_data, test_data, vocabulary_size

  def _read_subreddits(self, json_path, subreddit):
    with tf.gfile.GFile(json_path, "r") as f:
      data = json.load(f)
    all_words = data["subreddit_tokens"][subreddit]
    return [str(w) for w in all_words]

  def _load_subreddits_vocab(self, data_path):
    vocab_file = os.path.join(data_path, "vocab.json")
    with tf.gfile.GFile(vocab_file, "r") as f:
      vocab = json.load(f)
    word_to_id = vocab["token_to_id"]
    return word_to_id

  def _read_words(self, filename):
    with tf.gfile.GFile(filename, "r") as f:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()

  def _build_vocab(self, data):
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

  def _word_to_word_ids(self, words, word_to_id):
    # The vocab contains one out-of-vocab token.
    out_of_vocab_id = len(word_to_id)
    word_ids = []
    for word in words:
      word_ids.append(word_to_id.get(word, out_of_vocab_id))
    return word_ids


class WordLSTM(object):
  """Word-level LSTM model.

  Attributes:
    model_size: Number of parameters for the LSTM model, including word
      embedding and the softmax output layer.
  """

  def __init__(self,
               var_scope,
               is_training=True,
               config=None,
               data=None,
               reuse=tf.AUTO_REUSE,
               initializer=None):
    self._is_training = is_training
    # self._data is one instance of TextDataBatch()
    self._data = data
    self._config = config
    # The scale to initialize the vars using
    # tf.random_uniform_initializer().
    self._init_scale = config.init_scale
    self.batch_size = batch_size = data.batch_size
    self.num_steps = data.num_steps
    self.vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    embedding_size = config.embedding_size
    vocab_size = config.vocab_size
    self.model_size = embedding_size * vocab_size + 4 * (
        hidden_size * (hidden_size + embedding_size + 1) + vocab_size *
        (hidden_size + 1))

    with tf.variable_scope(
        utils.get_model_name_scope(var_scope),
        reuse=reuse,
        initializer=initializer):
      with tf.device("/cpu:0"):
        embedding = tf.get_variable(
            "embedding", [vocab_size, embedding_size], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, data.input)

      if is_training and config.keep_prob < 1:
        inputs = tf.nn.dropout(inputs, config.keep_prob)

      output, state, rnn_vars = self._build_rnn_graph(inputs, is_training)
      self._final_state = state

      softmax_w = tf.get_variable(
          "softmax_w", [vocab_size, hidden_size], dtype=tf.float32)
      softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)

      if FLAGS.model_split == 0:
        self._shared_vars = [embedding] + rnn_vars
        self._personal_vars = [softmax_w, softmax_b]
      elif FLAGS.model_split == 1:
        self._shared_vars = [embedding]
        self._personal_vars = rnn_vars + [softmax_w, softmax_b]
      else:
        raise ValueError("Unknown model splitting strategy: {}!".format(
            FLAGS.model_split))

      if config.num_samples > 0:
        samped_softmax_inputs = tf.reshape(data.target,
                                           [tf.size(data.target), 1])
        sampled_softmax_lstm_output = tf.reshape(output, [-1, hidden_size])
        loss_ = tf.nn.sampled_softmax_loss(
            softmax_w,
            softmax_b,
            samped_softmax_inputs,
            sampled_softmax_lstm_output,
            config.num_samples,
            vocab_size,
            partition_strategy="div",
            name="sampled_loss")
        loss_ /= tf.cast(self.batch_size, tf.float32)
      else:
        logits = tf.nn.xw_plus_b(output, tf.transpose(softmax_w), softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss_.
        logits = tf.reshape(logits, [batch_size, self.num_steps, vocab_size])
        # Use the contrib sequence loss_ and average over the batches.
        loss_ = tf.contrib.seq2seq.sequence_loss(
            logits,
            data.target,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

      self._loss = tf.reduce_sum(loss_)
      self._all_vars = self._shared_vars + self._personal_vars
      self._var_dict = utils.get_var_dict(self._all_vars)

      # Perplexity of the model will be updated along with the method:
      # self.run_one_epoch()
      self.perplexity = None
      self.perplexity_placeholder = tf.placeholder(tf.float32, [])
      self.perplexity_summary = tf.summary.scalar(utils.LOSS_SUMMARY_NAME,
                                                  self.perplexity_placeholder)

      if is_training:
        self._learning_rate = tf.Variable(0.0, trainable=False)
        self._new_learning_rate = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")

        self._learning_rate_update = tf.assign(self._learning_rate,
                                               self._new_learning_rate)

        if FLAGS.optimizer == "sgd":
          optimizer_all_var = tf.train.GradientDescentOptimizer(
              self._learning_rate)
          optimizer_shared_var = tf.train.GradientDescentOptimizer(
              self._learning_rate)
          optimizer_personal_var = tf.train.GradientDescentOptimizer(
              self._learning_rate)
        elif FLAGS.optimizer == "adam":
          optimizer_all_var = tf.train.AdamOptimizer(config.adam_learning_rate)
          optimizer_shared_var = tf.train.AdamOptimizer(
              config.adam_learning_rate)
          optimizer_personal_var = tf.train.AdamOptimizer(
              config.adam_learning_rate)
        else:
          raise ValueError("unknown optimizer: {}!".format(FLAGS.optimizer))

        self._train_op_all = self._generate_train_op(
            self._all_vars, optimizer_all_var, config.max_grad_norm)
        self._train_op_shared = self._generate_train_op(
            self._shared_vars, optimizer_shared_var, config.max_grad_norm)
        self._train_op_personal = self._generate_train_op(
            self._personal_vars, optimizer_personal_var, config.max_grad_norm)

        self.train_op_dict = {
            utils.VARS_TYPE_ALL: self.train_op_all,
            utils.VARS_TYPE_SHARED: self.train_op_shared,
            utils.VARS_TYPE_PERSONAL: self.train_op_personal
        }

  def _get_lstm_cell(self, config, is_training):
    """Set LSTM with options: BasicLSTMCell and Block."""
    if config.rnn_mode == BASIC:
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size,
          forget_bias=0.0,
          state_is_tuple=True,
          reuse=not is_training)
    elif config.rnn_mode == BLOCK:
      return tf.contrib.rnn.LSTMBlockCell(config.hidden_size, forget_bias=0.0)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

  def _build_rnn_graph(self, inputs, is_training):
    """Build the inference graph using canonical LSTM cells."""
    config = self._config

    def make_cell():
      cell = self._get_lstm_cell(config, is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(config.batch_size, tf.float32)
    state = self._initial_state

    # Before unstack, inputs shape is [batch_size, num_steps, embedding_size]
    rnn_scope = "RNN"
    inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    outputs, state = tf.nn.static_rnn(
        cell, inputs, initial_state=self._initial_state, scope=rnn_scope)

    rnn_full_scope = utils.add_suffix(rnn_scope, tf.get_variable_scope().name)
    rnn_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=rnn_full_scope)

    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state, rnn_vars

  def assign_lr(self, session, lr_value):
    session.run(
        self._learning_rate_update,
        feed_dict={self._new_learning_rate: lr_value})

  def export_handles_to_collections(self, name):
    """Exports model handles to collections."""
    self._name = name
    ops = {utils.add_prefix(self._name, "cost"): self._loss}
    if self._is_training:
      ops.update(
          learning_rate=self._learning_rate,
          new_learning_rate=self._new_learning_rate,
          learning_rate_update=self._learning_rate_update,
          train_op_all=self._train_op_all,
          train_op_personal=self._train_op_personal,
          train_op_shared=self._train_op_shared)

    for name, op in ops.iteritems():
      tf.add_to_collection(name, op)

    self._initial_state_name = utils.add_prefix(self._name, "initial")
    self._final_state_name = utils.add_prefix(self._name, "final")
    export_state_tuples(self._initial_state, self._initial_state_name)
    export_state_tuples(self._final_state, self._final_state_name)

  def import_handles_from_collections(self):
    """Imports model handles from collections."""
    if self._is_training:
      self._train_op_all = tf.get_collection_ref("train_op_all")[0]
      self._train_op_shared = tf.get_collection_ref("train_op_shared")[0]
      self._train_op_personal = tf.get_collection_ref("train_op_personal")[0]
      self._learning_rate = tf.get_collection_ref("learning_rate")[0]
      self._new_learning_rate = tf.get_collection_ref("new_learning_rate")[0]
      self._learning_rate_update = tf.get_collection_ref(
          "learning_rate_update")[0]

    self._loss = tf.get_collection_ref(utils.add_prefix(self._name, "cost"))[0]
    if self._name == "Train":
      num_replicas = max(1, FLAGS.num_gpus)
    else:
      num_replicas = 1
    self._initial_state = import_state_tuples(
        self._initial_state, self._initial_state_name, num_replicas)
    self._final_state = import_state_tuples(
        self._final_state, self._final_state_name, num_replicas)

  @property
  def data(self):
    return self._data

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._loss

  @property
  def loss(self):
    """Used to unify the API for different models.

    This is the value that will be recorded in TB summaries.

    Returns:
      perplexity of current model.
    """
    return self.perplexity

  @property
  def loss_placeholder(self):
    return self.perplexity_placeholder

  @property
  def loss_summary(self):
    return self.perplexity_summary

  @property
  def final_state(self):
    return self._final_state

  @property
  def learning_rate(self):
    return self._learning_rate

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name

  @property
  def all_vars(self):
    return self._all_vars

  @property
  def shared_vars(self):
    return self._shared_vars

  @property
  def personal_vars(self):
    return self._personal_vars

  @property
  def var_dict(self):
    return self._var_dict

  @property
  def train_op_all(self):
    return self._train_op_all

  @property
  def train_op_shared(self):
    return self._train_op_shared

  @property
  def train_op_personal(self):
    return self._train_op_personal

  def create_saver(self):
    return tf.train.Saver()

  def _generate_train_op(self, vars_, optimizer, max_grad_norm):
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(self._loss, vars_), max_grad_norm)

    train_op = optimizer.apply_gradients(zip(grads, vars_))

    return train_op

  def run_one_epoch(self,
                    sess,
                    verbose=False,
                    update_vars_type=utils.VARS_TYPE_ALL):
    """Modifies and returns the perplexity."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = sess.run(self.initial_state)

    fetches = {
        "cost": self.cost,
        "final_state": self.final_state,
    }

    if self._is_training:
      fetches["train_op"] = self.train_op_dict[update_vars_type]

    print("training: {}, num_batches: {}".format(self._is_training,
                                                 self.data.num_batches))
    for step in range(self.data.num_batches):
      feed_dict = self.data.get_batch_feed_dict(batch_id=step)
      for i, (c, h) in enumerate(self.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h

      vals = sess.run(fetches, feed_dict)
      cost = vals["cost"]
      state = vals["final_state"]

      costs += cost
      iters += self.data.num_steps

      if verbose and step % (self.data.num_batches // 5) == 0:
        print("%.3f perplexity: %.3f speed: %.0f wps" %
              (step * 1.0 / self.data.num_batches, np.exp(costs / iters),
               iters * self.data.batch_size * max(1, FLAGS.num_gpus) /
               (time.time() - start_time)))

    self.perplexity = np.exp(costs / iters)
    return self.perplexity

  def add_perplexity_summary(self, sess, writer, global_step):
    """Adds perplexity summary.

    Args:
      sess: TF session.
      writer: File writer.
      global_step: Indicates global epoch id for all clients.
    """
    perplexity = self.run_one_epoch(sess, verbose=False)
    perplexity_summary = sess.run(
        self.perplexity_summary,
        feed_dict={self.perplexity_placeholder: perplexity})

    writer.add_summary(perplexity_summary, global_step=global_step)

  def get_summary_feed_dict(self):
    return {self.perplexity_placeholder: self.perplexity}
