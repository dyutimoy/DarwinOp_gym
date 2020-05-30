from gym.envs.registration import register

register(
    id='DarwinBulletEnv-v0',
    entry_point='darwin_gym.envs:DarwinBulletEnv',
)
