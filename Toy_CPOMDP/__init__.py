from gym.envs.registration import register
register(
    id='cpomdp-v0',
    entry_point='Toy_CPOMDP.envs:Toy_Env',
)