import os
import time
import tensorflow as tf
from six.moves import range
from logging import getLogger

logger = getLogger(__name__)

def get_model_dir(config, exceptions=None):
  keys = dir(config)
  keys.sort()
  keys.remove('env_name')
  keys = ['env_name'] + keys

  names = [config.env_name]
  for key in keys:
    # Only use useful flags
    if key not in exceptions:
      value = getattr(config, key)
      names.append(
        "%s=%s" % (key, ",".join([str(i) for i in value])
                   if type(value) == list else value))

  return os.path.join('checkpoints', *names) + '/'

def timeit(f):
  def timed(*args, **kwargs):
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()

    logger.info("%s : %2.2f sec" % (f.__name__, end_time - start_time))
    return result
  return timed
