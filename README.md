# Deep Reinforcement Learning in TensorFlow

TensorFlow implementation of Deep Reinforcement Learning papers. This implementation contains:

[1] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[2] [Human-Level Control through Deep Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)  
[3] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[4] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
[5] [Deep Exploration via Bootstrapped DQN](http://arxiv.org/abs/1602.04621) (in progress)  
[6] [Deep Recurrent Q-Learning for Partially Observable MDPs](http://arxiv.org/abs/1507.06527) (in progress)  
[7] [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783) (in progress)  
[8] [Continuous Deep q-Learning with Model-based Acceleration](http://arxiv.org/abs/1603.00748) (in progress)  


## Requirements

- Python 2.7
- [gym](https://github.com/openai/gym)
- [tqdm](https://github.com/tqdm/tqdm)
- [OpenCV2](http://opencv.org/) or [Scipy](https://www.scipy.org/)
- [TensorFlow](https://www.tensorflow.org/)


## Usage

First, install prerequisites with:

    $ pip install -U gym[all] tqdm scipy

Train with DQN model described in [[1]](#deep-reinforcement-learning-in-tensorflow):

    $ python main.py --network_header_type=nips --env_name=BreakOut-v0

Train with DQN model described in [[2]](#deep-reinforcement-learning-in-tensorflow):

    $ python main.py --network_header_type=nature --env_name=BreakOut-v0

Train with Double DQN model described in [[3]](#deep-reinforcement-learning-in-tensorflow):

    $ python main.py --agent_type=DDQN --env_name=BreakOut-v0

Train with Deuling network with Double Q-learning described in [[4]](#deep-reinforcement-learning-in-tensorflow):

    $ python main.py --agent_type=DDQN --network_output_type=duel --env_name=BreakOut-v0

Train with MLP model described in [[4]](#deep-reinforcement-learning-in-tensorflow) with corridor environment for debugging:

    $ python main.py --network_header_type=mlp --observation_dims='[16]' --env_name=CorridorSmall-v5 --display=True

To see all training options, run:

    $ python main.py --help

which will print:

    usage: main.py [-h] [--data_format DATA_FORMAT] [--agent_type AGENT_TYPE]
                  [--network_header_type NETWORK_HEADER_TYPE]
                  [--network_output_type NETWORK_OUTPUT_TYPE]
                  [--env_name ENV_NAME] [--n_action_repeat N_ACTION_REPEAT]
                  [--max_random_start MAX_RANDOM_START]
                  [--history_length HISTORY_LENGTH] [--max_r MAX_R]
                  [--min_r MIN_R] [--observation_dims OBSERVATION_DIMS]
                  [--random_start [RANDOM_START]] [--norandom_start]
                  [--preprocess [PREPROCESS]] [--nopreprocess]
                  [--is_train [IS_TRAIN]] [--nois_train] [--max_delta MAX_DELTA]
                  [--min_delta MIN_DELTA] [--ep_start EP_START] [--ep_end EP_END]
                  [--batch_size BATCH_SIZE] [--max_grad_norm MAX_GRAD_NORM]
                  [--memory_size MEMORY_SIZE] [--discount_r DISCOUNT_R]
                  [--t_ep_end T_EP_END] [--t_learn_start T_LEARN_START]
                  [--t_test T_TEST] [--t_train_max T_TRAIN_MAX]
                  [--t_train_freq T_TRAIN_FREQ]
                  [--t_target_q_update_freq T_TARGET_Q_UPDATE_FREQ]
                  [--learning_rate LEARNING_RATE]
                  [--learning_rate_minimum LEARNING_RATE_MINIMUM]
                  [--learning_rate_decay LEARNING_RATE_DECAY]
                  [--learning_rate_decay_step LEARNING_RATE_DECAY_STEP]
                  [--decay DECAY] [--momentum MOMENTUM] [--gamma GAMMA]
                  [--beta BETA] [--display [DISPLAY]] [--nodisplay]
                  [--log_level LOG_LEVEL] [--random_seed RANDOM_SEED] [--tag TAG]

    optional arguments:
      -h, --help            show this help message and exit
      --data_format DATA_FORMAT
                            The format of convolutional filter. NHWC for CPU and
                            NCHW for GPU
      --agent_type AGENT_TYPE
                            The type of agent [DQN, DDQN]
      --network_header_type NETWORK_HEADER_TYPE
                            The type of network header [mlp, nature, nips]
      --network_output_type NETWORK_OUTPUT_TYPE
                            The type of network output [normal, dueling]
      --env_name ENV_NAME   The name of gym environment to use
      --n_action_repeat N_ACTION_REPEAT
                            The number of actions to repeat
      --max_random_start MAX_RANDOM_START
                            The maximum number of NOOP actions at the beginning of
                            an episode
      --history_length HISTORY_LENGTH
                            The length of history of observation to use as an
                            input to DQN
      --max_r MAX_R         The maximum value of clipped reward
      --min_r MIN_R         The minimum value of clipped reward
      --observation_dims OBSERVATION_DIMS
                            The dimension of gym observation
      --random_start [RANDOM_START]
                            Whether to start with random state
      --norandom_start
      --preprocess [PREPROCESS]
                            Whether to preprocess the observation of environment
      --nopreprocess
      --is_train [IS_TRAIN]
                            Whether to do training or testing
      --nois_train
      --max_delta MAX_DELTA
                            The maximum value of delta
      --min_delta MIN_DELTA
                            The minimum value of delta
      --ep_start EP_START   The value of epsilon at start in e-greedy
      --ep_end EP_END       The value of epsilnon at the end in e-greedy
      --batch_size BATCH_SIZE
                            The size of batch for minibatch training
      --max_grad_norm MAX_GRAD_NORM
                            The maximum gradient norm of RMSProp optimizer
      --memory_size MEMORY_SIZE
                            The size of experience memory
      --discount_r DISCOUNT_R
                            The discount factor for reware
      --t_ep_end T_EP_END   The time when epsilon reach ep_end
      --t_learn_start T_LEARN_START
                            The time when to begin training
      --t_test T_TEST       The maximum number of t while training
      --t_train_max T_TRAIN_MAX
                            The maximum number of t while training
      --t_train_freq T_TRAIN_FREQ
      --t_target_q_update_freq T_TARGET_Q_UPDATE_FREQ
      --learning_rate LEARNING_RATE
                            The learning rate of training
      --learning_rate_minimum LEARNING_RATE_MINIMUM
                            The learning rate of training
      --learning_rate_decay LEARNING_RATE_DECAY
                            The learning rate of training
      --learning_rate_decay_step LEARNING_RATE_DECAY_STEP
                            The learning rate of training
      --decay DECAY         Decay of RMSProp optimizer
      --momentum MOMENTUM   Momentum of RMSProp optimizer
      --gamma GAMMA         Discount factor of return
      --beta BETA           Beta of RMSProp optimizer
      --display [DISPLAY]   Whether to do display the game screen or not
      --nodisplay
      --log_level LOG_LEVEL
                            Log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]
      --random_seed RANDOM_SEED
                            Value of random seed
      --tag TAG             The name of tag for a model, only for debugging


## Results

(in progress)


## References

- [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)
- [DeepMind's code](https://sites.google.com/a/deepmind.com/dqn/)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
