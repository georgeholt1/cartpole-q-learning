import gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm


class QAgent:
    """Q-learning agent for cart-pole environment.
    
    Parameters
    ----------
    n_bins_pos, n_bins_vel, n_bins_ang, n_bins_ang_vel : int, optional
        Number of bin edges in the state space variables position, velocity,
        angle and angular velocity.
        Defaults to 5.
    lr_init, eps_init : float, optional
        Initial value of learning rate and epsilon.
        Defaults to 1.0.
    lr_min, eps_min : float, optional
        Minimum value of learning rate and epsilon.
        Defaults to 0.05.
    lr_decay, eps_decay : float, optional
        Decay factor of learning rate and epsilon.
        `lr_decay` deafults to 10, and `eps_decay` to 20.
    discount_factor : float, optional
        Defaults to 0.9.
    n_episodes : int, optional
        Number of episodes for training.
        Defaults to 500.
    progress_bars : bool, optional
        Whether to show tqdm progress bar per episode.
        Defaults to True.
    Q_table_path : str, optional
        Path to a pre-populated Q-table. No checks are performed to see if
        the table is compatible with the other parameters.
        Defaults to None, which initialises a Q-table of the correct shape with
        zeros.
    track_Q_tables : bool, optional
        If set to True, the Q-table at the end of each episode during training
        is appended to a list of Q-tables.
        Defaults to False 
        
    Attributes
    ----------
    env : Gym CartPole environment
    Q_table : numpy array
        Q table. Initialised with zeros or loaded from file and updated during
        training.
    steps : list
        List of number of steps that the pole was kept upright corresponding
        to episode. Populated during training.
    avg_steps : float
        Average steps over training.
    learn_area : int
        Area under episodic learning curve.
    states : list
        List of states at each iteration when the agent is run. Populated during
        a call to `run_agent()`.
    tracked_Q_tables : list
        List of Q-tables at the end of each episode. Populated during training
        if `track_Q_talbes` is set to `True`.
    """
    def __init__(
        self,
        n_bins_pos=3, n_bins_vel=2, n_bins_ang=5, n_bins_ang_vel=8,
        lr_init=1.0, lr_min=0.05, lr_decay=41,
        eps_init=1.0, eps_min=0.05, eps_decay=26,
        discount_factor=0.96,
        n_episodes=2000,
        progress_bars=True,
        Q_table_path=None,
        track_Q_tables=False
    ):
        self.n_bins_pos = n_bins_pos
        self.n_bins_vel = n_bins_vel
        self.n_bins_ang = n_bins_ang
        self.n_bins_ang_vel = n_bins_ang_vel
        self.lr_init = lr_init
        self.lr = lr_init
        self.lr_min = lr_min
        self.lr_decay = lr_decay
        self.eps_init = eps_init
        self.eps = eps_init
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.discount_factor = discount_factor
        self.n_episodes = n_episodes
        self.progress_bars = progress_bars
        self.Q_table_path = Q_table_path
        self.track_Q_tables = track_Q_tables
        
        self.rng = default_rng()
        self.env = gym.make('CartPole-v1')
        self.bins = self._create_bins()
        self.steps = []
        self.avg_steps = None
        self.states = None
        self.tracked_Q_tables = []
        
        if self.Q_table_path is None:
            self.Q_table = self._initialise_Q_table()
        else:
            self.Q_table = self._load_Q_table()
        
        
    def _initialise_Q_table(self):
        """Create Q-table of correct shape filled with zeros."""
        return np.zeros(
            (self.n_bins_pos, self.n_bins_vel, self.n_bins_ang,
             self.n_bins_ang_vel, self.env.action_space.n)
        )
    
    
    def _load_Q_table(self):
        """Load pre-populated Q-table from file.
        
        File should have been created with the save_Q_table() method.
        """
        Q_table = np.load(self.Q_table_path)
        return Q_table
        
        
    def _reset(self):
        self.lr = self.lr_init
        self.eps = self.eps_init
        self.Q_table = self._initialise_Q_table()
        self.tracked_Q_tables = []
        self.steps = []
        self.avg_steps = None
        self.states = None
        
    
    def _create_bins(self):
        """Create bins for discretising the state space.
        
        Returns
        -------
        bins : dict
            Dictionary where keys are the state variable names and values
            are lists of bin edges.
        """
        bins = {
            'position': np.linspace(
                self.env.observation_space.low[0],
                self.env.observation_space.high[0],
                self.n_bins_pos
            ),
            'velocity': np.linspace(-5, 5, self.n_bins_vel),
            'angle': np.linspace(
                self.env.observation_space.low[2],
                self.env.observation_space.high[2],
                self.n_bins_ang
            ),
            'angular_velocity': np.linspace(-5, 5, self.n_bins_ang_vel)
        }        
        return bins
    
    
    def _discretise_state(self, state):
        """Discretise a state by binning the values.
        
        Outliers are considered to be in the nearest edge bin.
        
        Parameters
        ----------
        state : list
            List containing observation of cart-pole state.
        
        Returns
        -------
        tuple
            4-tuple containing bin index of discretised value for each state
            variable.
        """
        # Unpack state observations
        pos_obs, vel_obs, ang_obs, ang_vel_obs = state
        
        # Discretise each one
        def discretise_observation(obs, var):
            discr_obs = np.digitize(
                np.clip(obs, self.bins[var].min(), self.bins[var].max()),
                self.bins[var]
            ) - 1
            return discr_obs
        pos_discr = discretise_observation(pos_obs, 'position')
        vel_discr = discretise_observation(vel_obs, 'velocity')
        ang_discr = discretise_observation(ang_obs, 'angle')
        ang_vel_discr = discretise_observation(ang_vel_obs, 'angular_velocity')
        
        return (pos_discr, vel_discr, ang_discr, ang_vel_discr)
    
    
    def _update_Q_table(self, old_state, new_state, action, reward):
        """Update the Q table using Q-learning algorithm.
        
        Parameters
        ----------
        old_state, new_state : tuple
            Discretised 4-tuple state descriptor.
        action : int
            Action relating `old_state` to `new_state`.
        reward : int
        
        References
        ----------
        - Watkins, C. J. C. H. "Learning from Delayed Rewards" (1989)
          https://www.academia.edu/3294050/Learning_from_delayed_rewards
        - Sutton, R. S. & Barto, A. G. "Reinforcement Learning: An Introduction"
          2nd ed. (2018). Eqn. (6.8).
        """
        self.Q_table[old_state][action] += self.lr * (reward + \
            self.discount_factor * np.max(self.Q_table[new_state]) - \
            self.Q_table[old_state][action])
        
        
    def _epsilon_greedy_action(self, state):
        """Select best guess of optimal action with probability (1 - epsilon).
        Otherwise, sample random action from space of all possible actions.
        
        Returns
        -------
        int
        """
        if self.rng.uniform() > self.eps:
            return np.argmax(self.Q_table[state])
        else:
            return self.env.action_space.sample()
        
        
    def _update_learning_rate(self, episode):
        """Update the learning rate to decay with increasing episode number."""
        new_lr = max(
            self.lr_min,
            min(self.lr_init, 1-np.log10((episode+1)/self.lr_decay))
        )
        return new_lr
        
        
    def _update_epsilon(self, episode):
        """Update the epsilon value to decay with increasing episode number."""
        new_eps = max(
            self.eps_min,
            min(self.eps_init, 1-np.log10((episode+1)/self.eps_decay))
        )
        return new_eps
    
    
    def _calculate_performance_metrics(self):
        """Calculate a couple of performance metrics.
        
        Returns
        -------
        avg_steps : float
            Average number of steps before termination.
        """
        avg_steps = np.average(self.steps)
        return avg_steps
        
        
    def train(self):
        """Train the Q-learning agent."""
        
        episode_gen = range(self.n_episodes)
        if self.progress_bars:
            episode_gen = tqdm(episode_gen)
        
        for episode in episode_gen:
            step = 0
            state = self._discretise_state(self.env.reset())
            done = False
            while not done:
                step += 1
                action = self._epsilon_greedy_action(state)
                new_state, reward, done, _ = self.env.step(action)
                new_state = self._discretise_state(new_state)
                self._update_Q_table(state, new_state, action, reward)
                state = new_state
            
            # Record how well this episode went
            self.steps.append(step)
            
            if self.track_Q_tables:
                self.tracked_Q_tables.append(self.Q_table)
            
            # Update learning rate and epsilon
            self.lr = self._update_learning_rate(episode)
            self.eps = self._update_epsilon(episode)
        
        self.env.close()
        
        # Performance metrics
        self.avg_steps = self._calculate_performance_metrics()
        
        
    def save_Q_table(self, path):
        """Save a copy of the current Q-table.
        
        Parameters
        ----------
        path : str
            Path to file to save the Q-table to.
        """
        np.save(path, self.Q_table)
        
        
    def plot_lr_eps(
        self,
        ax=None,
        ax_labels=True,
        legend=True,
        grid=True,
        lr_plot_kwargs={'label': 'Learning rate', 'c': 'C0', 'ls': '-'},
        eps_plot_kwargs={'label': 'Epsilon', 'c': 'C1', 'ls': '--'},
        show=True
    ):
        """Plot learning rate and epsilon as a function of episode number."""
        lr_plot = [self.lr_init]
        eps_plot = [self.eps_init]
        for e in range(1, self.n_episodes):
            lr_plot.append(self._update_learning_rate(e))
            eps_plot.append(self._update_epsilon(e))
        
        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(lr_plot, **lr_plot_kwargs)
        ax.plot(eps_plot, **eps_plot_kwargs)
        if ax_labels:
            ax.set_xlabel('Episode')
            ax.set_ylabel(r'LR or $\epsilon$')
        if legend:
            ax.legend()
        if grid:
            ax.grid()
        ax.set_xlim(0, self.n_episodes)
        ax.set_ylim(0, max(self.lr_init, self.eps_init))

        if show:
            fig.show()
        
        
    def plot_progress(
        self, ax=None, ax_labels=True, grid=True, plot_kwargs={}, show=True
    ):
        """Plot the learning progress of the agent."""
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.steps, **plot_kwargs)
        if ax_labels:
            ax.set_xlabel('Episode')
            ax.set_ylabel('Steps')
        if grid:
            ax.grid()
        ax.set_xlim(0, self.n_episodes)
        ax.set_ylim(0, 500)

        if show:
            fig.show()
        
        
    def run_agent(self, render=True):
        """Run the agent with the current Q-table and no stochasticity
        in the action sampling.
        
        Parameters
        ----------
        render : bool, optional
            Defaults to True.
        """
        state = self._discretise_state(self.env.reset())
        state_discr = self._discretise_state(state)
        done = False
        self.states = [state]
        while not done:
            if render:
                self.env.render()
            action = np.argmax(self.Q_table[state_discr])
            state, reward, done, _ = self.env.step(action)
            self.states.append(state)
            state_discr = self._discretise_state(state)


    def plot_states(self, axs=None, alpha=0.5, lw=0.7):
        """Plot the values of the state variables over the course of an episode.

        Only makes sense to call this after a call to `run_agent()`, otherwise
        `states` will be empty.
        
        Parameters
        ----------
        axs : list, optional
            List of axes on which to draw the data. Should be length 4 and in
            the order: position, velocity, angle, angular velocity. If `None`,
            then a new figure is created.
            Defaults to None.
        alpha : float, optional
            Line opacity.
            Defaults to 0.5.
        lw : float, optional
            Line width.
            Defaults to 0.7.

        Returns
        -------
        The figure object and axes list, if `axs=None`. Otherwise returns
        nothing.
        """
        if axs is None:
            ret = True
            fig, axs = plt.subplots(2, 2, constrained_layout=True)
            axs = axs.flat
            axs[0].set_ylabel('Position')
            axs[1].set_ylabel('Velocity')
            axs[2].set_ylabel('Angle')
            axs[3].set_ylabel('Angular velocity')
            axs[2].set_xlabel('Time step')
            axs[3].set_xlabel('Time step')
            for ax in axs:
                ax.set_xlim(0, 500)
            axs[0].set_ylim(self.bins['position'][0], self.bins['position'][-1])
            axs[1].set_ylim(self.bins['velocity'][0], self.bins['velocity'][-1])
            axs[2].set_ylim(self.bins['angle'][0], self.bins['angle'][-1])
            axs[3].set_ylim(
                self.bins['angular_velocity'][0],
                self.bins['angular_velocity'][-1]
            )
        else:
            ret = False

        axs[0].plot(np.array(self.states)[:, 0], c='C0', alpha=alpha, lw=lw)
        axs[1].plot(np.array(self.states)[:, 1], c='C1', alpha=alpha, lw=lw)
        axs[2].plot(np.array(self.states)[:, 2], c='C2', alpha=alpha, lw=lw)
        axs[3].plot(np.array(self.states)[:, 3], c='C3', alpha=alpha, lw=lw)

        if ret:
            return fig, axs