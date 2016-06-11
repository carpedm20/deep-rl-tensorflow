import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .history import History
from .experience import Experience

class DoubleDeepQ(BaseModel):
  def __init__(self, network, environment, sess):
    self.sess = sess

    self.network = network
    self.target_network = network.copy()

    self.env = environment
    self.history = History(self.config)
    self.experience = Experience(self.config, self.model_dir)

  def train(self):
    start_step = self.stats.get_step()
    screen, reward, action, terminal = self.env.new_random_game()

    for _ in range(self.history_length):
      self.history.add(screen)

    for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
      ep = self.ep_end + max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t)

      # 1. predict
      action = self.predict(self.history.get(), ep)
      # 2. act
      screen, reward, terminal = self.env.act(action, is_training=True)
      # 3. observe
      self.observe(screen, reward, action, terminal)

      if terminal:
        screen, reward, action, terminal = self.env.new_random_game()

      if self.step >= self.learn_start and self.callback:
        self.callback.on_step(action, reward, terminal, exploration_rate)

  def predict(self, s_t, ep):
    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      action = self.network.predict([s_t])[0]
    return action

  def observe(self, screen, reward, action, terminal):
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.history.add(screen)
    self.experience.add(screen, reward, action, terminal)

    if self.step > self.learn_start:
      if self.step % self.train_frequency == 0:
        self.q_learning_mini_batch()

      if self.step % self.target_q_update_step == self.target_q_update_step - 1:
        self.update_target_q_network()

  def q_learning_mini_batch(self):
    if self.experience.count < self.history_length:
      return
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
      self.learning_rate_step: self.step,
    })

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})
