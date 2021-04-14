from gym.envs.registration import register

register(
    id= 'Airline-v0',
    entry_point='n_server.envs.n_server:NServerEnv',
)