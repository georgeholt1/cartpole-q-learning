# Cart-Pole Q-Learning
A space for experimenting with applying the Q-learning algorithm to the
cart-pole environment.

The Q-learning agent and associated methods are located in `src/QAgent.py`. The
`cart-pole.ipynb` notebook walks through usage of the `QAgent` class, performs a
hyperparameter search, and shows the behaviour of an agent enacting a policy
that gives good performance (it 'solves' the problem, i.e. it consistently
reaches 500 time steps).

## Requirements
Tested with Python 3.9.12.

The following sequence of commands sets up the environment.

```
conda create --name q-learning-cart-pole python=3.9
conda activate q-learning-cart-pole
conda install -c conda-forge jupyter numpy matplotlib pandas tqdm
python -m pip install pygame
python -m pip install gym
```
