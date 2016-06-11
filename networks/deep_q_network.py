import os
import tensorflow as tf

from .ops import conv2d, linear, batch_sample

class Network(object):
  def __init__(self, sess, data_format, history_length,
               screen_height, screen_width,
               action_size, activation_fn=tf.nn.relu,
               initializer=tf.truncated_normal_initializer(0, 0.02), 
               gamma=0.01, beta=0.0, global_network=None, global_optim=None, DQN_type=''):
    self.sess = sess

    if data_format == 'NHWC':
      self.inputs = tf.placeholder('float32',
          [None, screen_width, screen_height, history_length], name='inputs')
    elif data_format == 'NCHW':
      self.inputs = tf.placeholder('float32',
          [None, history_length, screen_width, screen_height], name='inputs')
    else:
      raise ValueError("unknown data_format : %s" % data_format)

    if data_format == 'NCHW':
      device = '/gpu:0'
    elif data_format == 'NHWC':
      device = '/cpu:0'
    else:
      raise ValueError('Unknown data_format: %s' % data_format)

    self.var = {}
    self.l0 = tf.div(self.inputs, 255.)

    if DQN_type.lower() == 'nature':
      with tf.variable_scope('Nature_DQN'), tf.device(device):
        self.l1, self.var['l1_w'], self.var['l1_b'] = conv2d(self.l0,
            32, [8, 8], [4, 4], initializer, activation_fn, data_format, name='l1_conv')
        self.l2, self.var['l2_w'], self.var['l2_b'] = conv2d(self.l1,
            64, [4, 4], [2, 2], initializer, activation_fn, data_format, name='l2_conv')
        self.l3, self.var['l3_w'], self.var['l3_b'] = conv2d(self.l2,
            64, [3, 3], [1, 1], initializer, activation_fn, data_format, name='l3_conv')
        self.l4, self.var['l4_w'], self.var['l4_b'] = \
            linear(self.l3, 512, activation_fn=activation_fn, name='l4_linear')
        self.outputs = self.l4
    elif DQN_type.lower() == 'nips':
        self.l1, self.var['l1_w'], self.var['l1_b'] = conv2d(self.l0,
            16, [8, 8], [4, 4], initializer, activation_fn, data_format, name='l1_conv')
        self.l2, self.var['l2_w'], self.var['l2_b'] = conv2d(self.l1,
            32, [4, 4], [2, 2], initializer, activation_fn, data_format, name='l2_conv')
        self.l3, self.var['l3_w'], self.var['l3_b'] = \
            linear(self.l2, 256, activation_fn=activation_fn, name='l3_linear')
        self.outputs = self.l3
    else:
      raise ValueError('Wrong DQN type: %s' % DQN_type)
