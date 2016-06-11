import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .agent import Agent
from .history import History
from .experience import Experience

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

    self.min_reward = conf.min_reward
    self.max_reward = conf.max_reward

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

  def train(self, saver, model_dir, t_max):
    tf.initialize_all_variables().run()
    self.pred_network.load_model(saver, model_dir)

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
      q, loss = self.observe(observation, reward, action, terminal)

      #if self.t >= self.t_learn_start and self.stat:
      #  self.stat.on_step(action, reward, terminal, ep, q, loss)

      if terminal:
        observation, reward, terminal = self.new_game()

  def predict(self, s_t, ep):
    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      action = self.pred_network.predict([s_t])[0]
    return action

  def observe(self, observation, reward, action, terminal):
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.history.add(observation)
    self.experience.add(observation, reward, action, terminal)

    result = 0, 0
    if self.t > self.t_learn_start:
      if self.t % self.t_train_freq == 0:
        result = self.q_learning_mini_batch()

      if self.t % self.t_target_q_update_freq == self.t_target_q_update_freq - 1:
        self.update_target_q_network()

    return result

  def q_learning_mini_batch(self):
    if self.experience.count < self.history_length:
      return 0, 0
    else:
      s_t, action, reward, s_t_plus_1, terminal = self.experience.sample()

    t = time.time()
    q_t_plus_1 = self.target_network.predict(s_t_plus_1)

    terminal = np.array(terminal) + 0.
    max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
    target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

    _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.q_summary], {
      self.target_q_t: target_q_t,
      self.action: action,
      self.s_t: s_t,
      self.learning_rate_step: self.t,
    })

    return q_t, loss

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})
