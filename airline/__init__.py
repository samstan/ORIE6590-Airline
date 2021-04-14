from gym.envs.registration import register

register(
    id= 'Airline-v0',
    entry_point='airline.envs.airline:AirlineEnv',
)