import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from logging import getLogger

logger = getLogger(__name__)

class Agent(object):
  def train(self, t_max):
    tf.initialize_all_variables().run()

    self.stat.load_model()
    self.target_network.run_copy()

    start_t = self.stat.get_t()
    observation, reward, terminal = self.new_game()

    for _ in range(self.history_length):
      self.history.add(observation)

    for self.t in tqdm(range(start_t, t_max), ncols=70, initial=start_t):
      ep = (self.ep_end +
          max(0., (self.ep_start - self.ep_end)
            * (self.t_ep_end - max(0., self.t - self.t_learn_start)) / self.t_ep_end))

      # 1. predict
      action = self.predict(self.history.get(), ep)
      # 2. act
      observation, reward, terminal, info = self.env.step(action, is_training=True)
      # 3. observe
      q, loss, is_update = self.observe(observation, reward, action, terminal)

      logger.debug("a: %d, r: %d, t: %d, q: %.4f, l: %.2f" % \
          (action, reward, terminal, np.mean(q), loss))

      if self.stat:
        self.stat.on_step(self.t, action, reward, terminal,
                          ep, q, loss, is_update, self.learning_rate_op)
      if terminal:
        observation, reward, terminal = self.new_game()

  def predict(self, s_t, ep):
    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      action = self.pred_network.calc_actions([s_t])[0]
    return action

  def q_learning_minibatch_test(self):
    s_t = np.array([[[ 0., 0., 0., 0.],
                     [ 0., 0., 0., 0.],
                     [ 0., 0., 0., 0.],
                     [ 1., 0., 0., 0.]]], dtype=np.uint8)
    s_t_plus_1 = np.array([[[ 0., 0., 0., 0.],
                            [ 0., 0., 0., 0.],
                            [ 1., 0., 0., 0.],
                            [ 0., 0., 0., 0.]]], dtype=np.uint8)
    s_t = s_t.reshape([1, 1] + self.observation_dims)
    s_t_plus_1 = s_t_plus_1.reshape([1, 1] + self.observation_dims)

    action = [3]
    reward = [1]
    terminal = [0]

    terminal = np.array(terminal) + 0.
    max_q_t_plus_1 = self.target_network.calc_max_outputs(s_t_plus_1)
    target_q_t = (1. - terminal) * self.discount_r * max_q_t_plus_1 + reward

    _, q_t, a, loss = self.sess.run([
        self.optim, self.pred_network.outputs, self.pred_network.actions, self.loss
      ], {
        self.targets: target_q_t,
        self.actions: action,
        self.pred_network.inputs: s_t,
      })
    logger.debug("q: %s, a: %d, l: %.2f" % (q_t, a, loss))
