import tensorflow as tf

class Statistic(object):
  def __init__(self, sess, t_test):
    self.sess = sess
    self.t_test = t_test

    with tf.variable_scope('t'):
      self.t_op = tf.Variable(0, trainable=False, name='t')
      self.t_add_op = self.t_op.assign_add(1)

  def get_t(self):
    return self.t_op.eval(session=self.sess)

  def on_step(self, t, action, reward, terminal, exploration_rate, q, loss):
    if t % self.t_test == self.t_test - 1:
      if not terminal:
        num_game = 0
        total_reward = 0.
        self.total_loss = 0.
        self.total_q = 0.
        self.update_count = 0
        ep_reward = 0.
        ep_rewards = []
      else:
        avg_reward = self.total_reward / self.test_step
        avg_loss = self.total_loss / self.update_count
        avg_q = self.total_q / self.update_count

        try:
          max_ep_reward = np.max(ep_rewards)
          min_ep_reward = np.min(ep_rewards)
          avg_ep_reward = np.mean(ep_rewards)
        except:
          max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

        print '\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
            % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game)

        if max_avg_ep_reward * 0.9 <= avg_ep_reward:
          self.step_assign_op.eval({self.step_input: self.step + 1}, session=self.sess)
          self.save_model(self.step + 1)

          max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

        if self.step > 180:
          self.inject_summary({
              'average/reward': avg_reward,
              'average/loss': avg_loss,
              'average/q': avg_q,
              'episode/max reward': max_ep_reward,
              'episode/min reward': min_ep_reward,
              'episode/avg reward': avg_ep_reward,
              'episode/num of game': num_game,
              'episode/rewards': ep_rewards,
              'training/learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
            }, self.step)

    self.step_add_op.eval(session=self.sess)

  def inject_summary(self, tag_dict, step):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)
