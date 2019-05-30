from gym.envs.registration import register

register(
    id='FieldedMove-v0',
    entry_point='gym_fieldedmove.envs:FieldedMove',
)