import os
import time
import numpy as np
import tensorflow as tf
from logging import getLogger

from .agent import Agent
from .history import History
from .experience import Experience

logger = getLogger(__name__)

class DeepQ(Agent):
  def __init__(self, sess, pred_network, target_network, env, stat, conf):
    self.sess = sess
    self.stat = stat

    self.ep_start = conf.ep_start
    self.ep_end = conf.ep_end
    self.history_length = conf.history_length
    self.t_ep_end = conf.t_ep_end
    self.t_learn_start = conf.t_learn_start
    self.t_train_freq = conf.t_train_freq
    self.t_target_q_update_freq = conf.t_target_q_update_freq

    self.discount_r = conf.discount_r
    self.min_r = conf.min_r
    self.max_r = conf.max_r
    self.min_delta = conf.min_delta
    self.max_delta = conf.max_delta
    self.max_grad_norm = conf.max_grad_norm
    self.observation_dims = conf.observation_dims

    self.learning_rate = conf.learning_rate
    self.learning_rate_minimum = conf.learning_rate_minimum
    self.learning_rate_decay = conf.learning_rate_decay
    self.learning_rate_decay_step = conf.learning_rate_decay_step

    self.pred_network = pred_network
    self.target_network = target_network
    self.target_network.create_copy_op(self.pred_network)

    self.env = env 
    self.experience = Experience(conf.data_format,
        conf.batch_size, conf.history_length, conf.memory_size, conf.observation_dims)
    self.history = History(conf.data_format,
        conf.batch_size, conf.history_length, conf.observation_dims)

    if conf.random_start:
      self.new_game = self.env.new_random_game
    else:
      self.new_game = self.env.new_game

    # Optimizer
    with tf.variable_scope('optimizer'):
      self.targets = tf.placeholder('float32', [None], name='target_q_t')
      self.actions = tf.placeholder('int64', [None], name='action')

      actions_one_hot = tf.one_hot(self.actions, self.env.action_size, 1.0, 0.0, name='action_one_hot')
      pred_q = tf.reduce_sum(self.pred_network.outputs * actions_one_hot, reduction_indices=1, name='q_acted')

      self.delta = self.targets - pred_q
      if self.max_delta and self.min_delta:
        self.delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta')

      self.loss = tf.reduce_mean(tf.square(self.delta), name='loss')

      self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
          tf.train.exponential_decay(
              self.learning_rate,
              self.stat.t_op,
              self.learning_rate_decay_step,
              self.learning_rate_decay,
              staircase=True))

      optimizer = tf.train.RMSPropOptimizer(
        self.learning_rate_op, momentum=0.95, epsilon=0.01)
      
      grads_and_vars = optimizer.compute_gradients(self.loss)
      for idx, (grad, var) in enumerate(grads_and_vars):
        if grad is not None:
          grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
      self.optim = optimizer.apply_gradients(grads_and_vars)

  def observe(self, observation, reward, action, terminal):
    reward = max(self.min_r, min(self.max_r, reward))

    self.history.add(observation)
    self.experience.add(observation, reward, action, terminal)

    # q, loss, is_update
    result = [], 0, False

    if self.t > self.t_learn_start:
      if self.t % self.t_train_freq == 0:
        result = self.q_learning_minibatch()

      if self.t % self.t_target_q_update_freq == self.t_target_q_update_freq - 1:
        self.update_target_q_network()

    return result

  def q_learning_minibatch(self):
    if self.experience.count < self.history_length:
      return [], 0, False
    else:
      s_t, action, reward, s_t_plus_1, terminal = self.experience.sample()

    terminal = np.array(terminal) + 0.

    # Deep Q-learning
    max_q_t_plus_1 = self.target_network.calc_max_outputs(s_t_plus_1)
    target_q_t = (1. - terminal) * self.discount_r * max_q_t_plus_1 + reward

    _, q_t, loss = self.sess.run([self.optim, self.pred_network.outputs, self.loss], {
      self.targets: target_q_t,
      self.actions: action,
      self.pred_network.inputs: s_t,
    })

    return q_t, loss, True

  def update_target_q_network(self):
    self.target_network.run_copy()
