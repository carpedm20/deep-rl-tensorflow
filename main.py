import gym
import random
import logging
import tensorflow as tf

from utils import get_model_dir
from agents.deep_q import DeepQ
from networks.mlp import MLPSmall
from agents.statistic import Statistic
from environments.environment import ToyEnvironment

SCALE = 10000

flags = tf.app.flags

# Deep q Network
#flags.DEFINE_string('DQN_type', 'nips', 'The type of DQN in A3C model. [nature, nips]')
flags.DEFINE_string('data_format', 'NCHW', 'The format of convolutional filter. NHWC for CPU and NCHW for GPU')

# Environment
flags.DEFINE_string('env_name', 'CorridorSmall-v5', 'The name of gym environment to use')
flags.DEFINE_integer('n_action_repeat', 1, 'The number of actions to repeat')
flags.DEFINE_integer('max_random_start', 30, 'The maximum number of NOOP actions at the beginning of an episode')
flags.DEFINE_integer('history_length', 1, 'The length of history of observation to use as an input to DQN')
flags.DEFINE_integer('max_r', +1, 'The maximum value of clipped reward')
flags.DEFINE_integer('min_r', -1, 'The minimum value of clipped reward')
flags.DEFINE_string('observation_dims', '[81]', 'The dimension of gym observation')
flags.DEFINE_boolean('random_start', False, 'Whether to start with random state')
flags.DEFINE_boolean('preprocess', False, 'Whether to preprocess the observation of environment')

# Training
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_float('ep_start', 1., 'The value of epsilon at start in e-greedy')
flags.DEFINE_float('ep_end', 0.01, 'The value of epsilnon at the end in e-greedy')
flags.DEFINE_integer('batch_size', 32, 'The size of batch for minibatch training')
flags.DEFINE_integer('max_grad_norm', 40, 'The maximum gradient norm of RMSProp optimizer')
flags.DEFINE_integer('memory_size', 100 * SCALE, 'The size of experience memory')
flags.DEFINE_integer('discount_r', 0.99, 'The discount factor for reware')

# Timer
flags.DEFINE_integer('t_ep_end', 10 * SCALE, 'The time when epsilon reach ep_end')
flags.DEFINE_integer('t_learn_start', 0.1 * SCALE, 'The time when to begin training')
flags.DEFINE_integer('t_test', SCALE, 'The maximum number of t while training')
flags.DEFINE_integer('t_train_max', 1000 * SCALE, 'The maximum number of t while training')
flags.DEFINE_integer('t_train_freq', 4, '')
flags.DEFINE_integer('t_target_q_update_freq', SCALE, '')

# Optimizer
flags.DEFINE_float('learning_rate', 0.025, 'The learning rate of training')
flags.DEFINE_float('learning_rate_minimum', 0.0025, 'The learning rate of training')
flags.DEFINE_float('learning_rate_decay', 0.96, 'The learning rate of training')
flags.DEFINE_float('learning_rate_decay_step', 1000, 'The learning rate of training')
flags.DEFINE_float('decay', 0.99, 'Decay of RMSProp optimizer')
flags.DEFINE_float('momentum', 0.0, 'Momentum of RMSProp optimizer')
flags.DEFINE_float('gamma', 0.99, 'Discount factor of return')
flags.DEFINE_float('beta', 0.01, 'Beta of RMSProp optimizer')

# Debug
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_string('log_level', 'INFO', 'Log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

conf = flags.FLAGS

logger = logging.getLogger()
logger.propagate = False
logger.setLevel(conf.log_level)

# set random seed
tf.set_random_seed(conf.random_seed)
random.seed(conf.random_seed)

def main(_):
  conf.observation_dims = eval(conf.observation_dims)

  model_dir = get_model_dir(conf,
      ['log_level', 'max_random_start', 'n_worker', 'random_seed', 't_save', 't_train', 'display'])

  with tf.Session() as sess:
    env = ToyEnvironment(conf.env_name, conf.n_action_repeat, conf.max_random_start,
                      conf.observation_dims, conf.data_format, conf.display)

    pred_network = MLPSmall(sess=sess,
                            observation_dims=conf.observation_dims,
                            history_length=conf.history_length,
                            output_size=env.env.action_space.n,
                            hidden_sizes=[50, 50, 50],
                            hidden_activation_fn=tf.sigmoid,
                            output_activation_fn=None,
                            name='pred_network')
    target_network = MLPSmall(sess=sess,
                              observation_dims=conf.observation_dims,
                              history_length=conf.history_length,
                              output_size=env.env.action_space.n,
                              hidden_sizes=[50, 50, 50],
                              hidden_activation_fn=tf.sigmoid,
                              output_activation_fn=None, 
                              name='target_network', trainable=False)

    stat = Statistic(sess, conf.t_test, conf.t_learn_start, model_dir, pred_network.var.values())
    agent = DeepQ(sess, pred_network, target_network, env, stat, conf)

    agent.train(conf.t_train_max)

if __name__ == '__main__':
  tf.app.run()
