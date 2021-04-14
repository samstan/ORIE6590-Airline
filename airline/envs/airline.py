import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class AirlineEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        
        # Initializes model parameters based on a configuration dictionary
        self.A = config['A']
        self.f = config['f']
        self.P = config['P']
        self.starting_state = config['starting_state']

        # Defines state and action spaces, sets current state to be starting_state
        self.action_space = gym.spaces.MultiDiscrete([1]*len(self.A.shape[1]))
        self.observation_space = gym.spaces.MultiDiscrete(self.starting_state)
        self.state = np.asarray(self.starting_state)

    # Resets environment to initial state
    def reset(self):
        self.state = np.asarray(self.starting_state)
        return self.state

    # Defines one step of the MDP, returning the new state, reward, whether time horizon is finished, and a dictionary of information
    def step(self, action, t):
        
        # Just for personal double checking for Q learning algortihm, can ignore this
        # if self.state[0] == self.N and self.state[1] == self.N and action[0] == 0 and action[1] == 0:
            # print('Uh oh, stuck at absorbing state')
        
        
        trans = self.pr(self.state, action, t)
        states = list(trans.keys())
        probs = list(trans.values())
        # Computes new state
        newState = np.asarry(states[np.random.choice(range(len(states)), 1, p = probs)[0]])
        reward = self.r(self.state, newState)
        self.state = newState
        episode_over = False

        return self.state, reward, episode_over, {}



    # Auxilary function computing the reward
    def r(self, state, newState):
        if np.all(state == newState):
            return 0
        else:
            difInd = np.amax(state-newState)
            return f[difInd]

    # Auxilary function computing transition distribution
    def pr(self, state, action, t):
        transition_probs = {}
        for j in range(len(action)):
            nState = state - np.matmul(self.A[:, j ], action[j])
            if not np.all(nState == state):
                transition_probs[nState] == self.P[t, j]
        transition_probs[state] = 1 - sum(transition_probs.values())
        
        return transition_probs
