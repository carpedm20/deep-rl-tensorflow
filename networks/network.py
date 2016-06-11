import copy
import tensorflow as tf

class Network(object):
  def __init__(self, sess):
    self.sess = sess
    self.copy_op = None

  def predict(self, observation):
    return self.outputs.eval({self.inputs: observation})

  def run_copy(self):
    if self.copy_op is None:
      raise Exception("run `create_copy_op` first before copy")
    else:
      self.sess.run(self.copy_op)

  def create_copy_op(self, network):
    with tf.variable_scope('copy_from_target'):
      copy_ops = []

      for name in self.var.keys():
        copy_op = self.var[name].assign(network.var[name])
        copy_ops.append(copy_op)

      self.copy_op = tf.group(*copy_ops, name='copy_op')

  def save_model(self, saver, checkpoint_dir, step=None):
    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    saver.save(self.sess, checkpoint_dir, global_step=step)

  def load_model(self, saver, checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      fname = os.path.join(checkpoint_dir, ckpt_name)
      saver.restore(self.sess, fname)
      print(" [*] Load SUCCESS: %s" % fname)
      return True
    else:
      print(" [!] Load FAILED: %s" % checkpoint_dir)
      return False
