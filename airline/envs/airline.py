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
        self.tau = config['tau']
        self.starting_state = config['starting_state']

        # Defines state and action spaces, sets current state to be starting_state
        self.action_space = gym.spaces.MultiBinary(n =self.A.shape[1] )
        sstate = np.asarray(self.starting_state) + 1
        self.observation_space = gym.spaces.MultiDiscrete(sstate) 
        self.state = np.asarray(self.starting_state)
        self.timestep = 0

    # Resets environment to initial state
    def reset(self):
        self.state = np.asarray(self.starting_state) 
        self.timestep = 0
        return self.state

    # Defines one step of the MDP, returning the new state, reward, whether time horizon is finished, and a dictionary of information
    def step(self, action):
        
        # Just for personal double checking for Q learning algortihm, can ignore this
        # if self.state[0] == self.N and self.state[1] == self.N and action[0] == 0 and action[1] == 0:
            # print('Uh oh, stuck at absorbing state')
        trans = self.pr(self.state, action, self.timestep)
        states = list(trans.keys())
        probs = list(trans.values())
        # Computes new state
        newState = np.asarray(states[np.random.choice(range(len(states)), 1, p = probs)[0]])
        reward = self.r(self.state, newState, action)
        self.state = newState
        episode_over = False
        self.timestep += 1
        if self.timestep==self.tau:
            episode_over = True
        return self.state, reward, episode_over, {}



    # Auxilary function computing the reward
    def r(self, state, newState, action):
        if np.all(state == newState):
            return 0
        else:
            return self.f[np.argmax(action)]

    # Auxilary function computing transition distribution
    def pr(self, state, action, t):
        transition_probs = {}
        if np.sum(action) == 1:
            for j in range(len(action)):
                nState = np.copy(state) - self.A[:, j ]*action[j]
                if not np.all(nState == state) and len(nState[nState < 0]) == 0:
                    transition_probs[tuple(nState)] = self.P[t, j]
            transition_probs[tuple(state)] = 1 - sum(transition_probs.values())
        else:
            transition_probs[tuple(state)] = 1
        return transition_probs
