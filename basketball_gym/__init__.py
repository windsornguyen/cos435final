from basketball_gym.basketball_env import BasketballEnv
from gym.envs.registration import register

register(
    id='MuJoKobe-v0',
    entry_point='basketball_gym:BasketballEnv'
)