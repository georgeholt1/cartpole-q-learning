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
conda install -c conda-forge jupyter numpy matplotlib pandas tqdm seaborn
python -m pip install pygame gym
python -m pip install "ray[tune]"
```

If running on Apple Silicon, use [Miniforge](https://github.com/conda-forge/miniforge)
and the following commands instead.

```
conda create --name q-learning-cart-pole python=3.9
conda activate q-learning-cart-pole
python -m pip uninstall grpcio
conda install grpcio=1.43.0
conda install jupyter numpy matplotlib pandas tqdm
python -m pip install pygame gym seaborn
python -m pip install "ray[tune]"
```