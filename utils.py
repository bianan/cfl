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
"""Utility functions for manipulating variables in Federated personalization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


TRAIN_NAME = "Train"
VALIDATION_NAME = "Validation"
TEST_NAME = "Test"
LOSS_NAME = "loss"
LOSS_SUMMARY_NAME = "perplexity"

# Vars type.
VARS_TYPE_ALL = "all"
VARS_TYPE_SHARED = "shared"
VARS_TYPE_PERSONAL = "personal"


def get_train_name_scope(var_scope):
  return "/".join((var_scope, TRAIN_NAME))


def get_validation_name_scope(var_scope):
  return "/".join((var_scope, VALIDATION_NAME))


def get_test_name_scope(var_scope):
  return "/".join((var_scope, TEST_NAME))


def get_model_name_scope(var_scope):
  return "/".join((var_scope, "Model"))


def get_update_name_scope(var_scope):
  return "/".join((var_scope, "Update"))


def get_var_dict(vars_):
  """Gets a dict of var base_name (e.g. 'w') to the variable."""
  var_dict = {}
  for v in vars_:
    var_base_name = get_base_name(v)
    var_dict[var_base_name] = v
  return var_dict


def get_var_value_ops(var_dict):
  return {k: v.value() for k, v in var_dict.items()}


def get_base_name(var):
  return var.name.split("/")[-1].split(":")[0]


def get_update_name(var, var_scope):
  var_base_name = get_base_name(var)
  var_update_name = "update_%s_%s" % (var_scope, var_base_name)
  return var_update_name


def get_update_placeholder_name(var):
  var_base_name = get_base_name(var)
  placeholder_name = "placeholder_%s" % var_base_name
  return placeholder_name


def generate_update_ops(vars_):
  """Generates update ops and placeholders.

  For each var, it generates a placeholder to feed in the new values.
  Then it takes the mean of the inputs along dimension 0.

  Args:
    vars_: Vars for which the update ops will be generated.

  Returns:
    update_ops: A list of update ops.
    dict_update_placeholders: A dict of var base name to its update-placeholder.
  """

  update_ops = []
  dict_update_placeholders = {}
  for v in vars_:
    # For every var in the scope, add a placeholder to feed in the new values.
    # The placeholder may need to hold multiple values, this happens
    # when updating the server from many clients.
    var_in_shape = [None] + v.shape.as_list()
    var_in_name = get_update_placeholder_name(v)
    var_in = tf.placeholder(v.dtype, shape=var_in_shape, name=var_in_name)
    var_in_mean = tf.reduce_mean(var_in, 0)
    update_op = v.assign(var_in_mean)
    update_ops.append(update_op)
    dict_update_placeholders[get_base_name(v)] = var_in

  return update_ops, dict_update_placeholders


def print_vars_on_clients(clients, sess):
  for c in clients.values():
    print("client %d:" % c.id)
    print(sess.run(c.read_ops_all_vars))


def add_prefix(prefix, name):
  """Adds prefix to name."""
  return "/".join((prefix, name))


def add_suffix(suffix, name):
  """Adds subfix to name."""
  return "/".join((name, suffix))


def get_attribute_dict(class_instance):
  """Gets a dict of attributeds of a class instance."""
  # first start by grabbing the Class items
  attribute_dict = dict((x, y)
                        for x, y in class_instance.__class__.__dict__.items()
                        if x[:2] != "__")
  # then update the class items with the instance items
  attribute_dict.update(class_instance.__dict__)
  return attribute_dict
