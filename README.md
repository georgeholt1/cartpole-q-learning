# Cart-Pole Q-Learning
A space for experimenting with using the Q-learning algorithm with the cart-pole environment.

## Requirements
Tested with Python 3.9.12.

The following sequence of commands sets up the environment.

```
conda create --name q-learning-cart-pole python=3.9
conda activate q-learning-cart-pole
conda install -c conda-forge jupyter numpy matplotlib pandas tqdm
python -m pip install pygame gym ray
```

If running on Apple Silicon, use Miniforge and the following commands instead.

```
conda create --name q-learning-cart-pole python=3.9
conda activate q-learning-cart-pole
python -m pip uninstall grpcio
conda install grpcio=1.43.0
conda install jupyter numpy matplotlib pandas tqdm
python -m pip install pygame gym
python -m pip install "ray[tune]"
```
