from gym.envs.registration import register

register(
    id='FieldedMove-v0',
    entry_point='gym_fieldedmove.envs:FieldedMove',
)
register(
    id='FieldedMove-Meta-v0',
    entry_point='gym_fieldedmove.envs:FieldedMoveMeta',
)
register(
    id='FieldedMove-Arb-v0',
    entry_point='gym_fieldedmove.envs:FieldedMoveArb',
)