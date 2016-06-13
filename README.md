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

    $ python main.py --network_header_type=nips --env_name=Breakout-v0

Train with DQN model described in [[2]](#deep-reinforcement-learning-in-tensorflow):

    $ python main.py --network_header_type=nature --env_name=Breakout-v0

Train with Double DQN model described in [[3]](#deep-reinforcement-learning-in-tensorflow):

    $ python main.py --agent_type=DDQN --env_name=Breakout-v0

Train with Deuling network with Double Q-learning described in [[4]](#deep-reinforcement-learning-in-tensorflow):

    $ python main.py --agent_type=DDQN --network_output_type=duel --env_name=Breakout-v0

Train with MLP model described in [[4]](#deep-reinforcement-learning-in-tensorflow) with corridor environment for debugging:

    $ python main.py --network_header_type=mlp --network_output_type=normal --agent_type=DQN --observation_dims='[16]' --env_name=CorridorSmall-v5 --display=True --t_learn_start=0.1 --learning_rate_decay_step=0.1
    $ python main.py --network_header_type=mlp --network_output_type=normal --agent_type=DDQN --observation_dims='[16]' --env_name=CorridorSmall-v5 --display=True --t_learn_start=0.1 --learning_rate_decay_step=0.1
    $ python main.py --network_header_type=mlp --network_output_type=dueling --agent_type=DQN --observation_dims='[16]' --env_name=CorridorSmall-v5 --display=True --t_learn_start=0.1 --learning_rate_decay_step=0.1
    $ python main.py --network_header_type=mlp --network_output_type=dueling --agent_type=DDQN --observation_dims='[16]' --env_name=CorridorSmall-v5 --display=True --t_learn_start=0.1 --learning_rate_decay_step=0.1



## Results

Result of `Corridor-v5` in [[4]](#deep-reinforcement-learning-in-tensorflow) for DQN (purple), DDQN (red), Dueling DQN (green), Dueling DDQN (blue).

![model](assets/corridor_result.png)

(in progress)


## References

- [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)
- [DeepMind's code](https://sites.google.com/a/deepmind.com/dqn/)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
