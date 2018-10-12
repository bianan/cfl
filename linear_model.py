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
"""Linear model definition and its data generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
import tensorflow as tf
import utils

tf.app.flags.DEFINE_string('save_path', '/tmp/log/fl_simulation/linear_model',
                    'Summary output directory.')
FLAGS = tf.app.flags.FLAGS

LEARNING_RATE = 0.05
# Number of data points on a client.
CLIENT_DATA_LEN = 100
BATCH_SIZE = 10
# Fractions of data used for training and validation, respectively.
# Fraction of test data is: 1 - TRAIN_FRACTION - VALID_FRACTION.
TRAIN_FRACTION = 0.8
VALID_FRACTION = 0.1

# Regression data pair.
DataPairs = collections.namedtuple('DataPairs', ['inputs', 'targets'])


class LinearRegressionConfig(object):
  """Configs for the linear regression model."""

  class FirstConfig(object):
    """First trial of the configuration."""
    init_scale = 0.1
    learning_rate = 1e-3

  def __init__(self):
    self.train_config = self.FirstConfig()
    self.eval_config = self.FirstConfig()


class RegressionDataBatch(object):
  """Batch data generator.

  Attributes:
    batch_size: The standard batch size.
  """

  def __init__(self, data_pairs=None, name='Data'):
    self.batch_size = BATCH_SIZE
    self._inputs, self._targets = data_pairs.inputs, data_pairs.targets
    data_len = len(self._inputs)

    self.num_batches = data_len // self.batch_size
    self.input, self.target = self.batch_generator(name)

  def batch_generator(self, name):
    """Generates a batch of samples.

    Args:
      name: name scope.
    Returns:
      A tensor pair: x [batch_size, 1], y [batch_size, 1].
    """
    with tf.name_scope(name):
      inputs = tf.convert_to_tensor(
          self._inputs, name='inputs', dtype=tf.float32)
      targets = tf.convert_to_tensor(
          self._targets, name='targets', dtype=tf.float32)
      data_len = tf.size(inputs)
      num_batches = data_len // self.batch_size
      inputs = tf.reshape(inputs[0:self.batch_size * num_batches],
                          [self.batch_size, num_batches])
      targets = tf.reshape(targets[0:self.batch_size * num_batches],
                           [self.batch_size, num_batches])

      assertion = tf.assert_positive(num_batches, message='num_batches==0!')
      with tf.control_dependencies([assertion]):
        num_batches = tf.identity(num_batches, name='num_batches_in_an_epoch')

      i = tf.train.range_input_producer(
          limit=num_batches, shuffle=False).dequeue()
      x = tf.strided_slice(inputs, [0, i], [self.batch_size, i + 1])
      x.set_shape([self.batch_size, 1])
      y = tf.strided_slice(targets, [0, i], [self.batch_size, i + 1])
      y.set_shape([self.batch_size, 1])

      return x, y


class RegressionData(object):
  """Data generator for a specific client.

  Attributes:
    agent_id: Specifies the data distribution.
  """

  def __init__(self, configs=None, data_len=CLIENT_DATA_LEN, agent_id=0):
    self.agent_id = agent_id
    self.data_len = data_len
    (self.train_data, self.validation_data,
     self.test_data) = self._generate_data_pairs()

    self.train_data_batch = RegressionDataBatch(
        self.train_data, name=utils.TRAIN_NAME + 'Data')
    self.validation_data_batch = RegressionDataBatch(
        self.validation_data, name=utils.VALIDATION_NAME + 'Data')
    self.test_data_batch = RegressionDataBatch(
        self.test_data, name=utils.TEST_NAME + 'Data')

  def _generate_data_pairs(self):
    """Generates data pairs for linear model."""
    all_inputs = np.random.rand(self.data_len, 1)
    all_targets = 2. * all_inputs + self.agent_id

    num_train_data = int(self.data_len * TRAIN_FRACTION)
    num_validation_data = int(self.data_len * VALID_FRACTION)

    train_data = DataPairs(all_inputs[:num_train_data],
                           all_targets[:num_train_data])
    validation_data = DataPairs(
        all_inputs[num_train_data:(num_train_data + num_validation_data)],
        all_targets[num_train_data:(num_train_data + num_validation_data)])
    test_data = DataPairs(all_inputs[(num_train_data + num_validation_data):],
                          all_targets[(num_train_data + num_validation_data):])

    return train_data, validation_data, test_data


class LinearRegression(object):
  """Model class for linear regression.

  Attributes:
    data: An instance of data batch class.
    prediction: Prediction op generated by the model.
    all_vars: A list of all trainable vars.
    personal_vars: A list of personal vars that are trainable.
    shared_vars: A list of shared vars that are trainable.
    var_dict: A dict of var base name to var.
    loss: The loss measuring deviation between label and prediction.
    train_op_all: The train op over all vars.
    train_op_shared: The train op over shared vars.
    train_op_personal: The train op over personal vars.
  """

  def __init__(self,
               var_scope,
               is_training=True,
               data=None,
               config=None,
               reuse=tf.AUTO_REUSE,
               initializer=None):
    # Generates the model subgraph.
    with tf.variable_scope(utils.get_model_name_scope(var_scope), reuse=reuse):
      self.data = data
      w = tf.get_variable('w', shape=[1], trainable=is_training)
      b = tf.get_variable('b', shape=[1], trainable=is_training)
      self.prediction = w * data.input + b

      self.personal_vars = [b]
      self.shared_vars = [w]
      self.all_vars = self.personal_vars + self.shared_vars

    self.var_dict = utils.get_var_dict(self.all_vars)
    self.loss = tf.losses.mean_squared_error(data.target, self.prediction)
    self.loss_summary = tf.summary.scalar(utils.LOSS_NAME, self.loss)

    if is_training:
      opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
      self.train_op_all = opt.minimize(self.loss)
      self.train_op_shared = opt.minimize(self.loss, var_list=self.shared_vars)
      self.train_op_personal = opt.minimize(
          self.loss, var_list=self.personal_vars)

  def run_one_epoch(self,
                    sess,
                    verbose=False,
                    update_vars_type=utils.VARS_TYPE_ALL):
    pass

  @property
  def loss(self):
    """Used to unify the API for different models.

    This is the value that will be recorded in TensorBoard summaries.

    Returns:
      loss op.
    """
    return self.loss
